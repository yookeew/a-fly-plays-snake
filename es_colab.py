"""
ES entry point for the Colab T4 run (option 1: train the biophysics so the
retina obs is usable, then ask whether the real wiring beats its scrambles).

One arm per invocation, checkpointed to `--out` every generation so a reclaimed
Colab session resumes with `--resume` (the default here).

    python es_colab.py --arm synthetic --obs retina --generations 120 \
        --out /content/drive/MyDrive/flysnake/runs
    python es_colab.py --arm real   --obs retina --out .../runs
    python es_colab.py --arm rewire --obs retina --out .../runs

Arms (all share subgraph size / degree / E-I / readout / ports so the only
difference is *who wires to whom* -- the step-5 control design):
  real       -- FlyWire subgraph, `--hops` deep, retinotopic ports
  rewire     -- that subgraph, degree-preserving edge scramble
  synthetic  -- fresh random graph matched to the subgraph's n / in-degree /
                E-I / n_types (ports fall back to an even sensory split --
                retinotopy made no measurable difference even for real, see
                explore/retina_results.md)

`--obs feature` reproduces the step-5 setup (with the reflex + ES now) for a
sanity baseline.
"""
import argparse
import os

import numpy as np

from brain import load_flywire, make_synthetic, rewire_degree_preserving, _reach
from model import Brain
from snake import FEATURE_NAMES
from train import es_train, random_baseline


def build_cx(arm, obs, hops, n_readout, seed, full=False):
    n_obs = 3 * 7 * 7 if obs == "retina" else len(FEATURE_NAMES)
    ports = "retina" if obs == "retina" else "random"

    def _real():
        return load_flywire(n_obs=n_obs, hops=hops, full=full,
                            n_readout=n_readout, ports=ports, seed=seed)

    if arm == "real":
        return n_obs, _real()
    if arm == "rewire":
        return n_obs, rewire_degree_preserving(_real(), seed=seed)
    if arm == "synthetic":
        r = _real()
        cx = make_synthetic(
            n_obs, n_neurons=r.n, n_out=len(r.out_idx),
            mean_in_degree=max(1, round(r.W.nnz / r.n)),
            frac_inhib=1.0 - float((r.W.data > 0).mean()),
            n_types=r.n_types, seed=seed)
        return n_obs, cx
    raise ValueError(arm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["real", "rewire", "synthetic"])
    ap.add_argument("--obs", default="retina", choices=["retina", "feature"])
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--full", action="store_true",
                    help="whole connectome, no subgraph (slow; retina R2 showed "
                         "no gain over a subgraph -- benchmark before using)")
    ap.add_argument("--n-readout", type=int, default=64)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--board", type=int, default=12)
    ap.add_argument("--board-min", type=int, default=6,
                    help="curriculum start board; grows to --board. "
                         "set == --board to disable")
    ap.add_argument("--board-grow-every", type=int, default=12)
    ap.add_argument("--max-ticks", type=int, default=350)
    ap.add_argument("--inner-steps", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--shaping", type=float, default=1.0)
    ap.add_argument("--gain-init", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=".", help="checkpoint DIRECTORY")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--feature-warm-start", action="store_true",
                    help="feature obs only: use the danger-aware warm start")
    a = ap.parse_args()

    n_obs, cx = build_cx(a.arm, a.obs, a.hops, a.n_readout, a.seed, full=a.full)
    print(cx.summary(), flush=True)

    seeds = np.concatenate([np.asarray(p) for p in cx.ports])
    reach = _reach(cx.W, seeds, 20)[cx.out_idx].mean()
    print(f"port -> readout reachability (<=20 hops): {reach:.0%}", flush=True)

    rb_mean, rb_max = random_baseline(board=a.board)
    print(f"random policy (board {a.board}): mean {rb_mean:.2f}  max {rb_max}\n",
          flush=True)

    brain = Brain(cx, inner_steps=a.inner_steps, device=a.device)

    os.makedirs(a.out, exist_ok=True)
    sub = "full" if a.full else f"h{a.hops}"
    ckpt = os.path.join(
        a.out, f"{a.arm}_{a.obs}_{sub}_r{a.n_readout}_s{a.seed}.pkl")

    warm = a.obs == "feature" and a.feature_warm_start
    es_train(
        brain, generations=a.generations, pop=a.pop, sigma=a.sigma, lr=a.lr,
        n_envs=a.n_envs, board=a.board, board_min=a.board_min,
        board_grow_every=a.board_grow_every, max_ticks=a.max_ticks,
        shaping=a.shaping, gain_init=a.gain_init, warm_start=warm,
        readout_sigma_frac=0.25, obs_mode=a.obs, reflex=True, seed=a.seed,
        eval_every=10, out=ckpt, resume=not a.no_resume)
