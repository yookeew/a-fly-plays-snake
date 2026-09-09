"""
Deep R^2 probe: does retinotopy carry the food bearing to the descending
neurons when the WHOLE medulla->LC->DN pathway is in the graph (full=True)?

The quick pass (`runs/retina_iso_quick.log`, hops=2) had retinotopic
R2_sin=0.21 vs shuffled 0.09 -- a real ordering but far too weak, and only 56%
of the readout was even reachable. This asks whether DEPTH was the ceiling.

R^2 only -- no warm-start rollout, no food eval. `_env_targets(env)[:2]` is the
[food_sin, food_cos] a greedy rollout sees; regress centred descending activity
onto it, held-out. That is exactly the signal `warm_start_rollout` fits, so R^2
~ 0 means no readout wiring can rescue retina and the lever is ES on the
biophysical knobs instead.

retinotopic gets inner_steps {16, 28} x gain {2, 8}; the shuffled / random
controls get one config each. ~35 min at full size.
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from brain import load_flywire, _reach
from model import Brain
from train import _env_targets
from snake import SnakeEnv, greedy_bot

BOARD = 12
N_EP = 10


def r2_probe(cx, inner, gain, n_ep=N_EP, seed=1):
    """Held-out R^2, centred descending activity -> [food_sin, food_cos]."""
    brain = Brain(cx, inner_steps=inner)
    params = brain.unpack_pop(brain.init_params(seed=0, gain_init=gain)[None, :])
    R, Y = [], []
    for ep in range(n_ep):
        env = SnakeEnv(BOARD, BOARD, obs="retina", seed=seed + ep)
        o = env.reset()
        h = brain.initial_state_pop(1, 1)
        for _ in range(3 * BOARD * BOARD):
            Y.append(_env_targets(env)[:2])
            h, _ = brain.step_pop(
                h, o.reshape(1, 1, -1).astype(brain.dtype), params)
            r = np.clip(h[:, 0, :], 0.0, brain.r_max)[brain.out_idx]
            R.append((r[:, 0] - r[:, 0].mean()).copy())
            o, _, d = env.step(int(greedy_bot(env)))
            if d:
                break
    R, Y = np.array(R), np.array(Y)
    cut = len(R) * 3 // 4
    A = np.linalg.solve(R[:cut].T @ R[:cut] + 1e-2 * np.eye(R.shape[1]),
                        R[:cut].T @ Y[:cut])
    p = R[cut:] @ A
    ss_res = ((Y[cut:] - p) ** 2).sum(0)
    ss_tot = ((Y[cut:] - Y[cut:].mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot


def run(name, cx, configs):
    seeds = np.concatenate([np.asarray(p) for p in cx.ports])
    reach = _reach(cx.W, seeds, 20)[cx.out_idx].mean()
    print(f"\n{name}  n={cx.n}  ro_reach={reach:.0%}", flush=True)
    for inner, gain in configs:
        t0 = time.time()
        r2 = r2_probe(cx, inner, gain)
        print(f"  inner={inner:2d} gain={gain:.0f}  "
              f"R2(sin,cos)=({r2[0]:+.3f}, {r2[1]:+.3f})   "
              f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    N_RET = 3 * 7 * 7
    t0 = time.time()

    cx_ret = load_flywire(N_RET, ports="retina", full=True)
    print(f"loaded full retina  [{time.time() - t0:.0f}s]", flush=True)
    run("retinotopic [full]", cx_ret,
        [(16, 2.0), (16, 8.0), (28, 2.0), (28, 8.0)])

    cx_shuf = copy.copy(cx_ret)
    perm = np.random.default_rng(0).permutation(len(cx_ret.ports))
    cx_shuf.ports = [cx_ret.ports[k] for k in perm]
    run("shuffled [full]", cx_shuf, [(16, 4.0)])

    run("random [full]", load_flywire(N_RET, ports="random", full=True),
        [(16, 4.0)])

    print(f"\ntotal {time.time() - t0:.0f}s", flush=True)
