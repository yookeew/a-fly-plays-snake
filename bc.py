"""
Behaviour cloning: train the biophysical knobs + the readout by gradient descent
to imitate greedy_bot, instead of searching for reward with ES.

Why (PROJECT.md sec 4): the forward pass is differentiable w.r.t. the ~120
biophysical params and the readout -- only the argmax isn't. greedy_bot gives a
dense per-tick target, so there is no reward / exploration / warm-start-spike
problem. And unlike the linear warm start (which only fits the readout on frozen
activity), BC tunes the biophysics too, so the network's internal representation
can *become* linearly decodable -- the thing the retina R^2 probe showed is
missing.

W and the synaptic signs are never touched. The trained params are still ours.

Output: a `theta` in model.Brain's flat layout, so `evaluate()` / `watch.py`
just work on the result.

    # needs torch; run on Colab (GPU) or a torch venv
    python bc.py --arm real --obs retina --hops 3 --epochs 40 --out runs/
    python bc.py --arm synthetic --obs feature --epochs 30 --out runs/   # sanity
"""
import argparse
import os
import pickle
import time

import numpy as np

from brain import _reach
from model import Brain, N_ACT
from snake import SnakeEnv, greedy_bot, FEATURE_NAMES
from train import evaluate
from es_colab import build_cx


# --------------------------------------------------------------- teacher data

def collect(obs_mode, boards, eps_per_board, max_ticks, seed):
    """Roll greedy_bot; return a list of (obs (T, n_obs) f32, act (T,) i64)."""
    data = []
    k = 0
    for b in boards:
        for _ in range(eps_per_board):
            env = SnakeEnv(b, b, obs=obs_mode, seed=seed + k)
            k += 1
            o = env.reset()
            obs, acts = [], []
            for _ in range(max_ticks):
                a = int(greedy_bot(env))
                obs.append(o)
                acts.append(a)
                o, _, d = env.step(a)
                if d:
                    break
            data.append((np.asarray(obs, np.float32),
                         np.asarray(acts, np.int64)))
    return data


def make_batches(data, B, rng):
    """Pad episodes to a common length per batch; yield obs (B,T,n_obs),
    act (B,T), mask (B,T). Episodes are length-bucketed so padding waste stays
    small, then the batch order is shuffled."""
    by_len = sorted(range(len(data)), key=lambda i: len(data[i][1]))
    batches = [by_len[s:s + B] for s in range(0, len(by_len), B)]
    rng.shuffle(batches)
    for idx in batches:
        T = max(len(data[i][1]) for i in idx)
        n_obs = data[idx[0]][0].shape[1]
        obs = np.zeros((len(idx), T, n_obs), np.float32)
        act = np.zeros((len(idx), T), np.int64)
        mask = np.zeros((len(idx), T), np.float32)
        for r, i in enumerate(idx):
            o, a = data[i]
            obs[r, :len(a)] = o
            act[r, :len(a)] = a
            mask[r, :len(a)] = 1.0
        yield obs, act, mask


# ------------------------------------------------------------------- BC model

class BCBrain:
    """Differentiable one-tick forward over a frozen connectome. Per-type
    biophysics, so the trained params pack straight into model.Brain's theta.

    Truncated BC: `h` is detached between game ticks, so each backward covers
    one tick's `inner_steps` settling (plenty -- greedy_bot is ~reactive). A
    window > 1 is a later knob.
    """

    def __init__(self, cx, inner_steps=16, gain_init=2.0, r_max=1.0,
                 device="cpu", reg=1e-3, class_weight=None):
        import torch
        self.t = torch
        self.dev = device
        self.inner = inner_steps
        self.r_max = r_max
        self.reg = reg
        # greedy_bot goes straight ~76% of ticks; unweighted CE collapses to
        # "always straight" (CE ~0.72, food at the floor). Inverse-frequency
        # class weights force the readout to actually use the input to decide
        # when to turn.
        self.cw = (None if class_weight is None else
                   torch.as_tensor(class_weight, dtype=torch.float32,
                                   device=device))
        self.n = cx.n
        self.n_types = cx.n_types
        self.n_out = len(cx.out_idx)
        self.type_id = torch.as_tensor(cx.type_id, dtype=torch.long,
                                       device=device)
        self.out_idx = torch.as_tensor(np.asarray(cx.out_idx),
                                       dtype=torch.long, device=device)

        ref = Brain(cx)                       # reuse its P assembly + theta0
        self._t = ref._t
        self.n_params = ref.n_params

        def _csr(m):
            import scipy.sparse as sp
            m = sp.csr_matrix(m)
            return torch.sparse_csr_tensor(
                torch.from_numpy(m.indptr.astype(np.int64)),
                torch.from_numpy(m.indices.astype(np.int64)),
                torch.from_numpy(m.data.astype(np.float32)),
                size=m.shape, device=device)

        self.W = _csr(ref.W)
        self.P = _csr(ref.P)

        th0 = ref.init_params(seed=0, gain_init=gain_init)
        nt = self.n_types
        mk = lambda a: torch.tensor(a, dtype=torch.float32, device=device,
                                    requires_grad=True)
        self.raw_alpha = mk(th0[0:nt])
        self.raw_gain = mk(th0[nt:2 * nt])
        self.bias = mk(th0[2 * nt:3 * nt])
        self.W_out = mk(th0[self._t:self._t + N_ACT * self.n_out]
                        .reshape(N_ACT, self.n_out))
        self.b_out = mk(th0[self._t + N_ACT * self.n_out:])
        self._gain0 = self.raw_gain.detach().clone()
        self._alpha0 = self.raw_alpha.detach().clone()

    def params(self):
        return [self.raw_alpha, self.raw_gain, self.bias, self.W_out, self.b_out]

    def accumulate_grads(self, obs, act, mask):
        """obs (B,T,n_obs), act (B,T), mask (B,T) numpy. Forward the padded
        batch and backward ONCE PER TICK -- `h` is detached between ticks, so
        holding the whole episode's graph is both unnecessary and (at n~60k x
        inner~16 x T~250) a multi-GB OOM. Grads accumulate into .grad; the
        caller does zero_grad() before and step() after. Returns mean CE
        (float) for logging."""
        torch = self.t
        F = torch.nn.functional
        obs = torch.as_tensor(obs, device=self.dev)
        act = torch.as_tensor(act, device=self.dev)
        mask = torch.as_tensor(mask, device=self.dev)
        B, T, _ = obs.shape
        denom = float(mask.sum())
        if denom == 0:
            return 0.0

        h = torch.zeros((self.n, B), device=self.dev)
        total = 0.0
        n_correct = 0.0
        pred_hist = np.zeros(N_ACT)
        for k in range(T):
            # alpha/gain/bias recomputed each tick so this tick's backward
            # reaches the params; summed per-tick grads == grad of the sum.
            alpha = torch.sigmoid(self.raw_alpha)[self.type_id][:, None]
            gain = F.softplus(self.raw_gain)[self.type_id][:, None]
            bias = self.bias[self.type_id][:, None]
            drive = bias + torch.sparse.mm(self.P, obs[:, k, :].T)     # (n,B)
            hh = h
            for _ in range(self.inner):
                r = hh.clamp(0.0, self.r_max)
                rec = torch.sparse.mm(self.W, r)
                hh = (1 - alpha) * hh + alpha * (gain * rec + drive)
                hh = hh.clamp(-20.0, 20.0)                             # nan guard
            r = hh.clamp(0.0, self.r_max)
            r_out = r[self.out_idx]
            r_out = r_out - r_out.mean(0, keepdim=True)
            logits = self.W_out @ r_out + self.b_out[:, None]          # (3,B)
            ce = F.cross_entropy(logits.T, act[:, k], weight=self.cw,
                                 reduction="none")
            mk = mask[:, k]
            loss_k = (ce * mk).sum() / denom
            if loss_k.requires_grad:
                loss_k.backward()
            total += float(loss_k)
            with torch.no_grad():
                pred = logits.argmax(0)
                n_correct += float(((pred == act[:, k]).float() * mk).sum())
                for c in range(N_ACT):
                    pred_hist[c] += float(((pred == c).float() * mk).sum())
            h = hh.detach()

        reg = self.reg * ((self.raw_gain - self._gain0) ** 2
                          + (self.raw_alpha - self._alpha0) ** 2).sum()
        reg.backward()
        return total, n_correct, denom, pred_hist

    def to_theta(self):
        d = lambda x: x.detach().cpu().numpy()
        return np.concatenate([d(self.raw_alpha), d(self.raw_gain), d(self.bias),
                               d(self.W_out).ravel(), d(self.b_out)])


# ------------------------------------------------------------------- training

def train_bc(cx, obs_mode, *, inner_steps, gain_init, epochs, batch,
             lr, reg, balance, boards, eps_per_board, max_ticks, board_eval,
             device, seed, out):
    import torch

    rng = np.random.default_rng(seed)
    print("collecting greedy_bot rollouts...", flush=True)
    data = collect(obs_mode, boards, eps_per_board, max_ticks, seed)
    acts = np.concatenate([a for _, a in data])
    freq = np.bincount(acts, minlength=N_ACT) / len(acts)
    print(f"  {len(data)} episodes, {len(acts)} ticks, "
          f"mean {len(acts) / len(data):.0f}/ep   "
          f"greedy L/S/R {(freq * 100).round(1)}", flush=True)

    cw = None
    if balance:
        cw = (1.0 / np.maximum(freq, 1e-3))
        cw = cw / cw.mean()
        print(f"  class weights {cw.round(2)}", flush=True)

    bc = BCBrain(cx, inner_steps, gain_init, device=device, reg=reg,
                 class_weight=cw)
    opt = torch.optim.Adam(bc.params(), lr=lr)

    cpu_brain = Brain(cx, inner_steps=inner_steps)   # for honest eval
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        tl, nb, nc, nt = 0.0, 0, 0.0, 0.0
        ph = np.zeros(N_ACT)
        for obs, act, mask in make_batches(data, batch, rng):
            opt.zero_grad()
            loss, c, t, p = bc.accumulate_grads(obs, act, mask)
            torch.nn.utils.clip_grad_norm_(bc.params(), 5.0)
            opt.step()
            tl += loss
            nb += 1
            nc += c
            nt += t
            ph += p
        acc = nc / max(nt, 1)
        row = {"epoch": ep, "loss": tl / nb, "acc": acc,
               "pred": (ph / ph.sum()).round(2).tolist(),
               "sec": time.time() - t0}
        if ep % 5 == 0 or ep == 1:
            th = bc.to_theta()
            _, s0, x0 = evaluate(cpu_brain, th, 40, board_eval,
                                 3 * max_ticks, seed=99, obs_mode=obs_mode,
                                 reflex=False)
            _, s1, x1 = evaluate(cpu_brain, th, 40, board_eval,
                                 3 * max_ticks, seed=99, obs_mode=obs_mode,
                                 reflex=True)
            row.update(food=s0, food_max=x0, food_reflex=s1)
            print(f"ep {ep:3d}  {row['sec']:5.0f}s  loss {row['loss']:.3f}  "
                  f"acc {acc:.0%}  pred L/S/R {row['pred']}  "
                  f"food {s0:5.2f} (+reflex {s1:5.2f})  max {x0:.0f}", flush=True)
        else:
            print(f"ep {ep:3d}  {row['sec']:5.0f}s  loss {row['loss']:.3f}  "
                  f"acc {acc:.0%}  pred L/S/R {row['pred']}", flush=True)
        hist.append(row)
        if out:
            with open(out, "wb") as fh:
                pickle.dump({"theta": bc.to_theta(), "history": hist,
                             "cfg": dict(obs_mode=obs_mode, inner_steps=inner_steps,
                                         gain_init=gain_init)}, fh)
    return bc.to_theta(), hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["real", "rewire", "synthetic"])
    ap.add_argument("--obs", default="retina", choices=["retina", "feature"])
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--n-readout", type=int, default=64)
    ap.add_argument("--neurons", type=int, default=0)
    ap.add_argument("--inner-steps", type=int, default=16)
    ap.add_argument("--gain-init", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--balance", dest="balance", action="store_true",
                    default=True)
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    help="unweighted CE (collapses to greedy's 76%% straight)")
    ap.add_argument("--boards", type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--eps-per-board", type=int, default=40)
    ap.add_argument("--max-ticks", type=int, default=250)
    ap.add_argument("--board-eval", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=".", help="checkpoint DIRECTORY")
    a = ap.parse_args()

    device = a.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            raise SystemExit("bc.py needs torch")
    print(f"device: {device}", flush=True)

    _, cx = build_cx(a.arm, a.obs, a.hops, a.n_readout, a.seed, neurons=a.neurons)
    print(f"n={cx.n}  types={cx.n_types}  readout={len(cx.out_idx)}", flush=True)
    seeds = np.concatenate([np.asarray(p) for p in cx.ports])
    print(f"port->readout reach (<=20 hops): "
          f"{_reach(cx.W, seeds, 20)[cx.out_idx].mean():.0%}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    tag = f"bc_{a.arm}_{a.obs}_h{a.hops}_s{a.seed}"
    if a.neurons:
        tag += f"_n{a.neurons}"
    train_bc(cx, a.obs, inner_steps=a.inner_steps, gain_init=a.gain_init,
             epochs=a.epochs, batch=a.batch, lr=a.lr, reg=a.reg,
             balance=a.balance, boards=tuple(a.boards),
             eps_per_board=a.eps_per_board, max_ticks=a.max_ticks,
             board_eval=a.board_eval, device=device, seed=a.seed,
             out=os.path.join(a.out, tag + ".pkl"))
