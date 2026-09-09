"""
Step 3c: the model. A rate RNN over a fixed connectome.

numpy by default: ES only ever calls the forward pass, so autograd buys nothing
and torch would be an 800 MB install for no gain. `Brain(cx, device="cuda")`
switches the population forward pass (`unpack_pop` / `initial_state_pop` /
`step_pop` / `act_pop`, the only path ES exercises) onto a GPU via torch sparse
-- needed for ES on the retina obs, where the biophysics must be *trained* (a
frozen random draw carries no bearing) and a full-connectome CPU generation is
minutes. Everything else -- `step` / `act`, the warm starts, watch.py, the
checks -- stays numpy/CPU and needs no torch. Parity: `explore/torch_parity.py`.

The connectome (W, its signs, its sparsity) is FROZEN. What gets trained is a
handful of biophysical knobs the electron microscope cannot see -- one gain, one
time constant, one bias PER CELL TYPE (~34-40 numbers, not per-neuron) -- plus a
linear readout off the descending neurons. ~300 parameters total.

Dynamics, per game tick, iterated `inner_steps` times:

    r = clip(h, 0, r_max)
    h = (1 - alpha) * h  +  alpha * (gain * (W @ r) + bias + I)

with alpha = dt / tau in (0, 1) -- storing the leak rate directly instead of tau
keeps every update contractive, so a bad parameter draw cannot make h blow up
through the leak term alone (the recurrent term still can -- that is what the
step 3d stability sweep is for).

`inner_steps` must exceed the port -> readout hop count (4-6 in the real
subgraph) or the observation literally cannot reach the output in one tick.
Start at 6.
"""

import numpy as np
import scipy.sparse as sp

N_ACT = 3   # turn left / straight / turn right -- matches snake.ACTION_TURN


def _softplus(x):
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Brain:
    """A `Connectome` plus the trained parameter vector that animates it.

    Parameters are one flat float64 vector (what ES perturbs). Layout:

        [ raw_alpha | raw_gain | bias ]   each length n_types
        [ W_out (N_ACT x n_out) flattened | b_out (N_ACT) ]

        alpha = sigmoid(raw_alpha)          leak rate dt/tau, per type
        gain  = softplus(raw_gain)          recurrent gain, per type, > 0
        bias  = bias                        tonic drive, per type
    """

    def __init__(self, cx, r_max=1.0, dt=1.0, inner_steps=8, dtype=np.float32,
                 device="cpu"):
        self.cx = cx
        self.r_max = float(r_max)
        self.dt = float(dt)
        self.inner_steps = int(inner_steps)
        self.dtype = dtype
        self.device = device

        self.n = cx.n
        self.n_types = cx.n_types
        self.n_out = len(cx.out_idx)
        self.type_id = cx.type_id
        # float32: the sparse matvec is memory-bound, so halving the bytes very
        # nearly halves the wall time. ES does not need the precision.
        self.W = sp.csr_matrix(cx.W).astype(dtype)
        self.out_idx = np.asarray(cx.out_idx)

        # injection matrix P (n x n_obs): column c is 1 on the neurons of port c.
        # Then I = (P @ obs) broadcasts each observation channel onto its slab.
        rows = np.concatenate([np.asarray(p) for p in cx.ports])
        cols = np.concatenate([np.full(len(p), c) for c, p in enumerate(cx.ports)])
        if getattr(cx, "port_signs", None) is not None:
            vals = np.concatenate([np.asarray(s, float) for s in cx.port_signs])
        else:
            vals = np.ones(len(rows))
        self.P = sp.csr_matrix((vals, (rows, cols)),
                               shape=(self.n, cx.n_obs))

        self._t = 3 * self.n_types
        self.n_params = self._t + N_ACT * self.n_out + N_ACT

        if device != "cpu":
            self._to_torch(device)

    # ------------------------------------------------------------------- torch
    #
    # Minimal GPU path: only the batched population forward pass moves to the
    # device. `obs` arrives from the CPU envs as numpy and `logits` come back as
    # numpy so the caller (rollout_pop) does argmax / the collision reflex / the
    # env step exactly as before -- `h` is the one tensor that stays on device,
    # passed back opaquely each tick.

    def _to_torch(self, device):
        import torch

        self._torch = torch
        self._tdt = torch.float32

        def _csr(m):
            m = sp.csr_matrix(m)
            return torch.sparse_csr_tensor(
                torch.from_numpy(m.indptr.astype(np.int64)),
                torch.from_numpy(m.indices.astype(np.int64)),
                torch.from_numpy(m.data.astype(np.float32)),
                size=m.shape, device=device)

        self.W_t = _csr(self.W)
        self.P_t = _csr(self.P)
        self.out_idx_t = torch.as_tensor(self.out_idx, dtype=torch.int64,
                                         device=device)

    # ------------------------------------------------------------------ params

    def init_params(self, seed=0, gain_init=1.0, readout_scale=0.1):
        rng = np.random.default_rng(seed)
        theta = np.empty(self.n_params)
        # raw_alpha = 0  -> alpha 0.5 -> tau = 2*dt
        theta[0 * self.n_types:1 * self.n_types] = 0.0
        # invert softplus so gain starts at gain_init (swept in step 3d)
        theta[1 * self.n_types:2 * self.n_types] = np.log(np.expm1(gain_init))
        theta[2 * self.n_types:3 * self.n_types] = 0.0                 # bias
        tail = rng.normal(0.0, readout_scale, self.n_params - self._t)
        tail[-N_ACT:] = 0.0                                            # b_out
        theta[self._t:] = tail
        return theta

    def unpack(self, theta):
        nt = self.n_types
        alpha = _sigmoid(theta[0:nt])[self.type_id]          # (n,)
        gain = _softplus(theta[nt:2 * nt])[self.type_id]     # (n,)
        bias = theta[2 * nt:3 * nt][self.type_id]            # (n,)
        wt = theta[self._t:self._t + N_ACT * self.n_out].reshape(N_ACT, self.n_out)
        b_out = theta[self._t + N_ACT * self.n_out:]
        return alpha, gain, bias, wt, b_out

    # ----------------------------------------------------------------- forward

    def initial_state(self, batch):
        return np.zeros((batch, self.n))

    def step(self, h, obs, params):
        """One game tick. h (batch, n), obs (batch, n_obs) -> new h, logits (batch, N_ACT).

        `params` is the tuple from unpack(); pass it in so a batch of agents
        sharing one parameter vector unpacks it once, not P times.
        """
        alpha, gain, bias, wt, b_out = params
        I = (self.P @ obs.T).T                    # (batch, n), constant this tick
        drive = bias + I                          # broadcast bias (n,) over batch
        for _ in range(self.inner_steps):
            r = np.clip(h, 0.0, self.r_max)
            rec = (self.W @ r.T).T                # (batch, n); W[i,j] = j -> i
            h = (1.0 - alpha) * h + alpha * (gain * rec + drive)
        r = np.clip(h, 0.0, self.r_max)
        # centre the readout population before the linear map. Without this the
        # brightest descending neuron sets argmax and it never flips -- the
        # input-driven variation (~0.01) is dwarfed by the random per-neuron
        # offset. Centering makes the readout respond to the PATTERN of
        # descending activity, which is what we want anyway.
        r_out = r[:, self.out_idx]
        r_out = r_out - r_out.mean(axis=1, keepdims=True)
        logits = r_out @ wt.T + b_out
        return h, logits

    def act(self, h, obs, params):
        h, logits = self.step(h, obs, params)
        return h, np.argmax(logits, axis=1)

    # ------------------------------------------------- batched over a population
    #
    # ES evaluates 2*pop candidates per generation. They share W, so the one
    # expensive op -- the recurrent matvec -- should happen ONCE per inner step
    # for the whole population, not 2*pop times. State is (n, C, E): n neurons,
    # C candidates, E envs each. Reshapes to (n, C*E) are free (contiguous), so
    # `W @ state` stays a single sparse-times-dense call.

    def unpack_pop(self, thetas):
        """thetas (C, n_params) -> params tuple for step_pop.
        alpha/gain/bias (n, C, 1); wt (C, N_ACT, n_out); b_out (C, N_ACT, 1)."""
        C, nt, t = thetas.shape[0], self.n_types, self._t
        g = self.type_id
        dt = self.dtype
        alpha = _sigmoid(thetas[:, 0:nt])[:, g].T[:, :, None].astype(dt)
        gain = _softplus(thetas[:, nt:2 * nt])[:, g].T[:, :, None].astype(dt)
        bias = thetas[:, 2 * nt:3 * nt][:, g].T[:, :, None].astype(dt)
        wt = thetas[:, t:t + N_ACT * self.n_out].reshape(C, N_ACT, self.n_out)
        b_out = thetas[:, t + N_ACT * self.n_out:][:, :, None]
        # (1 - alpha) is used every inner step; fold it in here once.
        out = (alpha, (1.0 - alpha).astype(dt), gain, bias,
               wt.astype(dt), b_out.astype(dt))
        if self.device == "cpu":
            return out
        t_ = self._torch
        return tuple(t_.as_tensor(x, dtype=self._tdt, device=self.device)
                     for x in out)

    def initial_state_pop(self, C, E):
        if self.device == "cpu":
            return np.zeros((self.n, C, E), dtype=self.dtype)
        return self._torch.zeros((self.n, C, E), dtype=self._tdt,
                                 device=self.device)

    def step_pop(self, h, obs, params):
        """h (n, C, E), obs (C, E, n_obs) -> h, logits (C, N_ACT, E).

        In-place arithmetic in the inner loop -- at (n x C x E) every temporary
        array allocation shows up in the wall time. On device, `h` is a torch
        tensor passed straight back each tick; `logits` returns as numpy.
        """
        if self.device != "cpu":
            return self._step_pop_torch(h, obs, params)
        alpha, one_minus_alpha, gain, bias, wt, b_out = params
        n, C, E = h.shape
        I = (self.P @ obs.reshape(C * E, -1).T).reshape(n, C, E).astype(self.dtype)
        drive = bias + I
        r = np.empty_like(h)
        for _ in range(self.inner_steps):
            np.clip(h, 0.0, self.r_max, out=r)
            rec = (self.W @ r.reshape(n, C * E)).reshape(n, C, E)
            rec *= gain
            rec += drive
            rec *= alpha
            h *= one_minus_alpha
            h += rec
        np.clip(h, 0.0, self.r_max, out=r)
        r_out = r[self.out_idx]                          # (n_out, C, E)
        r_out = r_out - r_out.mean(axis=0, keepdims=True)
        logits = np.einsum("oce,cko->cke", r_out, wt) + b_out
        return h, logits

    def _step_pop_torch(self, h, obs, params):
        t_ = self._torch
        alpha, one_minus_alpha, gain, bias, wt, b_out = params
        n, C, E = h.shape
        obs_t = t_.as_tensor(np.ascontiguousarray(obs.reshape(C * E, -1).T),
                             dtype=self._tdt, device=self.device)
        drive = bias + t_.sparse.mm(self.P_t, obs_t).reshape(n, C, E)
        for _ in range(self.inner_steps):
            r = h.clamp(0.0, self.r_max).reshape(n, C * E)
            rec = t_.sparse.mm(self.W_t, r).reshape(n, C, E)
            h = one_minus_alpha * h + alpha * (gain * rec + drive)
        r = h.clamp(0.0, self.r_max)
        r_out = r[self.out_idx_t]                        # (n_out, C, E)
        r_out = r_out - r_out.mean(dim=0, keepdim=True)
        logits = t_.einsum("oce,cko->cke", r_out, wt) + b_out
        return h, logits.detach().cpu().numpy()

    def act_pop(self, h, obs, params):
        h, logits = self.step_pop(h, obs, params)
        return h, np.argmax(logits, axis=1)             # (C, E)


if __name__ == "__main__":
    # smoke test only -- the real reachability / stability / liveness checks are
    # step 3d. Here: does it run, keep shape, and not spew NaNs.
    import sys
    from brain import make_synthetic, load_flywire
    from snake import FEATURE_NAMES

    n_obs = len(FEATURE_NAMES)
    if len(sys.argv) > 1 and sys.argv[1] == "flywire":
        cx = load_flywire(n_obs=n_obs)
    else:
        cx = make_synthetic(n_obs=n_obs, seed=0)

    brain = Brain(cx, inner_steps=6)
    print(f"n={brain.n}  types={brain.n_types}  readout={brain.n_out}  "
          f"n_params={brain.n_params}")

    theta = brain.init_params(seed=0)
    params = brain.unpack(theta)
    rng = np.random.default_rng(0)

    batch = 8
    h = brain.initial_state(batch)
    acts = []
    for t in range(60):
        obs = rng.normal(0, 1, (batch, n_obs))
        h, a = brain.act(h, obs, params)
        acts.append(a)
    acts = np.stack(acts)
    print(f"mean|h| {np.abs(h).mean():.3f}   max|h| {np.abs(h).max():.3f}   "
          f"finite {np.isfinite(h).all()}")
    print(f"action counts over {acts.size} decisions: "
          f"{np.bincount(acts.ravel(), minlength=N_ACT)}")
