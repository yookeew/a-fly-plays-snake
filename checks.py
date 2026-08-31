"""
Step 3d: sanity checks, run BEFORE the game exists.

Broken dynamics still run, still print numbers, and still learn nothing. If you
skip this you will spend a day blaming the reward function for a network that
was never going to work.

Three checks, on either connectome:

  1. REACHABILITY  -- pulse the ports. Does activity actually arrive at the
     readout, and after how many inner steps? Must be < inner_steps.
  2. STABILITY     -- 100 ticks of random input. mean|h| must settle to a
     non-zero constant: not explode, not decay to nothing. This is where
     gain_init gets tuned, with no game in the loop to confuse things.
  3. READOUT LIVENESS -- do the three action logits actually vary, and is argmax
     not welded to one action? A net that always turns left passes 1 and 2 and
     is stone dead.

    python checks.py                 # synthetic
    python checks.py flywire         # real subgraph
"""

import sys

import numpy as np

from brain import make_synthetic, load_flywire, _reach
from model import Brain, N_ACT
from snake import FEATURE_NAMES

GAINS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def check_reachability(brain, params_at):
    print("\n[1] REACHABILITY  -- unit pulse on every port neuron, h starts at 0")

    seeds = np.concatenate([np.asarray(p) for p in brain.cx.ports])
    graph_hops = next(
        (h for h in range(1, 21)
         if _reach(brain.cx.W, seeds, h)[brain.out_idx].all()), None)
    print(f"    graph BFS: readout fully downstream of a port at {graph_hops} hops")

    obs = np.ones((1, brain.cx.n_obs))
    reached_at = None
    h = brain.initial_state(1)
    alpha, gain, bias, wt, b_out = params_at
    drive = bias + (brain.P @ obs.T).T
    for k in range(1, 3 * brain.inner_steps + 1):
        r = np.clip(h, 0.0, brain.r_max)
        h = (1 - alpha) * h + alpha * (gain * (brain.W @ r.T).T + drive)
        live = np.mean(np.abs(h[0, brain.out_idx]) > 1e-4)
        if reached_at is None and live > 0.5:
            reached_at = k
    print(f"    dynamics: >50% of readout active after {reached_at} inner steps "
          f"(inner_steps = {brain.inner_steps})")
    ok = reached_at is not None and reached_at <= brain.inner_steps
    print("    -> OK" if ok else "    -> FAIL: raise inner_steps or shrink the graph")
    return ok


def check_stability(cx):
    print("\n[2] STABILITY  -- 100 ticks of N(0,1) input per channel, sweep gain_init")
    print(f"    {'gain':>6}  {'mean|h| t=1':>12}  {'t=50':>10}  {'t=100':>10}  verdict")
    good = []
    for g in GAINS:
        brain = Brain(cx, inner_steps=6)
        theta = brain.init_params(seed=0, gain_init=g)
        params = brain.unpack(theta)
        rng = np.random.default_rng(0)
        h = brain.initial_state(4)
        traj = []
        for t in range(100):
            obs = rng.normal(0, 1, (4, cx.n_obs))
            h, _ = brain.step(h, obs, params)
            traj.append(np.abs(h).mean())
        t1, t50, t100 = traj[0], traj[49], traj[99]
        blow = (not np.isfinite(t100)) or t100 > 50
        drift = np.isfinite(t100) and abs(t100 - t50) / (t50 + 1e-9) > 0.25
        # "healthy" = settles AND at a magnitude the readout can actually see.
        # r is clipped to [0, r_max]; mean|h| well under ~0.05*r_max means the
        # signal that reaches the readout is nearly zero.
        lo, hi = 0.05 * brain.r_max, 3.0 * brain.r_max
        verdict = ("EXPLODES" if blow else "still drifting" if drift
                   else "settles LOW" if t100 < lo
                   else "settles HIGH" if t100 > hi else "healthy")
        if verdict == "healthy":
            good.append(g)
        print(f"    {g:>6.2f}  {t1:>12.4f}  {t50:>10.4f}  {t100:>10.4f}  {verdict}")
    pick = good[len(good) // 2] if good else None
    print(f"    -> gain_init = {pick}" if pick
          else "    -> FAIL: nothing settled; widen GAINS or check normalisation")
    return pick


def check_readout_liveness(cx, gain_init):
    print(f"\n[3] READOUT LIVENESS  -- gain_init={gain_init}, 200 ticks random input")
    brain = Brain(cx, inner_steps=6)
    best = None
    for seed in range(4):                       # a few random readouts
        theta = brain.init_params(seed=seed, gain_init=gain_init)
        params = brain.unpack(theta)
        rng = np.random.default_rng(100 + seed)
        h = brain.initial_state(8)
        logits = []
        for _ in range(200):
            obs = rng.normal(0, 1, (8, cx.n_obs))
            h, lg = brain.step(h, obs, params)
            logits.append(lg)
        lg = np.concatenate(logits)
        stds = lg.std(0)
        counts = np.bincount(lg.argmax(1), minlength=N_ACT)
        frac_top = counts.max() / counts.sum()
        alive = stds.min() > 1e-4 and frac_top < 0.95
        print(f"    seed {seed}: logit std {np.array2string(stds, precision=4)}  "
              f"argmax {counts}  {'alive' if alive else 'DEAD'}")
        best = best or alive
    print("    -> OK: logits vary and argmax is not stuck" if best
          else "    -> WEAK: ES may still rescue it, but bump readout_scale / gain")
    return best


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    n_obs = len(FEATURE_NAMES)
    cx = load_flywire(n_obs=n_obs) if which == "flywire" \
        else make_synthetic(n_obs=n_obs, seed=0)
    print(f"connectome: {which}")
    print(cx.summary())

    brain0 = Brain(cx, inner_steps=6)
    p0 = brain0.unpack(brain0.init_params(seed=0, gain_init=1.0))

    r_ok = check_reachability(brain0, p0)
    gain = check_stability(cx)
    live_ok = check_readout_liveness(cx, gain or 1.0)

    print(f"\nsummary: reachability {'ok' if r_ok else 'FAIL'} | "
          f"stable gain {gain} | readout {'ok' if live_ok else 'weak'}")
