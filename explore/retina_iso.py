"""
Task 1: is the `retina` input salvageable, and does retinotopy actually help?

`runs/iso.log` (pre-reflex) had retina+steer at ~0 food even after hops=3 lifted
port->readout reachability to 100% -- so DEPTH was not the whole story. This
re-runs the isolation with three things iso.log lacked:

  * the collision reflex as a survival floor (`evaluate(reflex=True)`), so a
    policy that steers badly still logs the food it stumbles into instead of
    dying on tick 20 -- separates "cannot steer" from "cannot survive".
  * a `random ports` control on the SAME retina obs. The question is whether the
    retinotopic ARRANGEMENT of the ports helps, not whether spatial input helps
    -- random 147 sensory neurons is the scramble.
  * full=True vs a hops=4 subgraph. full keeps the real medulla->LC->DN pursuit
    pathway even where it runs 6+ synapses deep; the subgraph rule can sever it
    or route the "reachable" path through unrelated shortcut interneurons.

Plus a diagnostic that decides the whole approach: the held-out R^2 of a ridge
fit from centred descending activity to [food_sin, food_cos]. That is exactly
the signal `warm_start_rollout` regresses. If R^2 ~ 0 on retina, the frozen
random-biophysics network does not turn a lit pixel into a bearing and no
readout wiring can fix it -- the lever is then ES on the biophysical knobs, not
the ports.

feature-obs / random ports is the reference line (~11 warm-start, ~20 +reflex).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from brain import load_flywire, _reach
from model import Brain
from train import warm_start_rollout, evaluate, _env_targets
from snake import FEATURE_NAMES, SnakeEnv, greedy_bot

K = 7
N_RET = 3 * K * K                 # 147
N_FEAT = len(FEATURE_NAMES)
BOARD = 12

# `quick` (default): one shallow hops=2 subgraph (~15k neurons, the size the old
#   iso.log used), 2 gains, 12 eps -- ~15 min, just "does the retinotopic
#   ARRANGEMENT beat random / shuffled ports at all".
# `full`: the real test -- full=True so the deep medulla->LC->DN pathway is
#   present even 6+ synapses in, inner_steps 16, 3 gains, 20 eps. ~90 min.
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
if MODE == "full":
    EP_EVAL, GAINS, INNER, WS_EPS = 20, (2.0, 4.0, 8.0), 16, 24
    RETINA_ARMS = (("full", dict(full=True)), ("h4", dict(hops=4)))
else:
    EP_EVAL, GAINS, INNER, WS_EPS = 12, (2.0, 4.0), 12, 16
    RETINA_ARMS = (("h2", dict(hops=2)),)


def bearing_r2(brain, theta, obs_mode, n_ep=16, seed=1):
    """Held-out R^2, descending activity -> [food_sin, food_cos], probed by
    rolling greedy_bot. Same signal warm_start_rollout fits; ~0 here means the
    readout cannot be wired no matter how."""
    params = brain.unpack_pop(theta[None, :])
    R, Y = [], []
    for ep in range(n_ep):
        env = SnakeEnv(BOARD, BOARD, obs=obs_mode, seed=seed + ep)
        o = env.reset()
        h = brain.initial_state_pop(1, 1)
        for _ in range(3 * BOARD * BOARD):
            Y.append(_env_targets(env)[:2])            # [sin, cos]
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
    pred = R[cut:] @ A
    ss_res = ((Y[cut:] - pred) ** 2).sum(0)
    ss_tot = ((Y[cut:] - Y[cut:].mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot                       # (2,)  sin, cos


def arm(name, cx, obs_mode):
    brain = Brain(cx, inner_steps=INNER)
    seeds = np.concatenate([np.asarray(p) for p in cx.ports])
    ro_reach = _reach(cx.W, seeds, 20)[cx.out_idx].mean()
    raw = rfx = -1.0
    r2 = np.array([np.nan, np.nan])
    for g in GAINS:
        th = warm_start_rollout(brain, brain.init_params(seed=0, gain_init=g),
                                obs_mode=obs_mode, n_episodes=WS_EPS,
                                boards=(BOARD,), seed=0)
        _, m0, _ = evaluate(brain, th, EP_EVAL, BOARD, 3 * BOARD * BOARD,
                            seed=99, obs_mode=obs_mode, reflex=False)
        _, m1, _ = evaluate(brain, th, EP_EVAL, BOARD, 3 * BOARD * BOARD,
                            seed=99, obs_mode=obs_mode, reflex=True)
        raw, rfx = max(raw, m0), max(rfx, m1)
        if g == GAINS[0]:
            r2 = bearing_r2(brain, th, obs_mode)
    print(f"{name:32s} n={cx.n:6d}  ro_reach={ro_reach:4.0%}  "
          f"R2(sin,cos)=({r2[0]:+.2f},{r2[1]:+.2f})  "
          f"food raw={raw:5.2f}  +reflex={rfx:5.2f}", flush=True)


if __name__ == "__main__":
    print(f"mode={MODE}  inner={INNER}  gains={GAINS}  eval_eps={EP_EVAL}",
          flush=True)
    arm("feature / random ports",
        load_flywire(N_FEAT, ports="random", seed=0), "feature")

    for tag, kw in RETINA_ARMS:
        cx_ret = load_flywire(N_RET, ports="retina", **kw)
        arm(f"retina / retinotopic [{tag}]", cx_ret, "retina")

        # shuffled: identical neuron groups, scrambled which pixel drives which.
        # Isolates the retinotopic ARRANGEMENT from "these particular medulla
        # cells". If retinotopic == shuffled, the layout is doing nothing.
        cx_shuf = copy.copy(cx_ret)
        perm = np.random.default_rng(0).permutation(len(cx_ret.ports))
        cx_shuf.ports = [cx_ret.ports[k] for k in perm]
        arm(f"retina / shuffled ports [{tag}]", cx_shuf, "retina")

        arm(f"retina / random ports [{tag}]",
            load_flywire(N_RET, ports="random", **kw), "retina")
