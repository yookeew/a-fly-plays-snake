"""#1 retry: route the food bearing through the fly's VISUAL PURSUIT channel
instead of olfaction.

func.log established that name-matched "functional" ports lose (food_* -> ORNs
gave 7.4 vs 11.4 for random ports) because olfaction is non-directional. Snake
food-seeking is really visual small-target pursuit, and the fly has dedicated
retinotopic hardware for exactly that (LC10/LC11/LC15/LC18 -> steer toward
target). This tries `ports="visual"`:

    food_sin  -> small-target LCs, +1 right eye / -1 left eye (push-pull bearing)
    food_cos  -> small-target LCs
    food_near -> small-target LCs
    danger_*  -> looming detectors LPLC2/LC4/LC6, by body side
    fullness  -> random central neurons

Readout kept at 64 random DNs (the config that scored best). Compared head to
head with random ports and the old olfactory functional ports.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from brain import load_flywire
from model import Brain
from train import warm_start_rollout, evaluate
from snake import FEATURE_NAMES

N_OBS = len(FEATURE_NAMES)
BOARD = 12
EPISODES = 40
GAINS = (2.0, 4.0, 8.0)
WS = (0, 1)


def arm(name, **kw):
    cx = load_flywire(n_obs=N_OBS, readout="random", n_readout=64, seed=0, **kw)
    reach = None
    brain = Brain(cx, inner_steps=8)
    best = -1.0
    for g in GAINS:
        runs = []
        for ws in WS:
            th = warm_start_rollout(brain, brain.init_params(seed=0, gain_init=g),
                                    obs_mode="feature", seed=ws)
            _, m, _ = evaluate(brain, th, EPISODES, BOARD, 3 * BOARD * BOARD, seed=99)
            runs.append(m)
        best = max(best, float(np.mean(runs)))
    print(f"{name:26s}  n={cx.n:6d}  best_food={best:5.2f}", flush=True)


arm("random ports",     ports="random")
arm("visual ports",     ports="visual")
arm("functional (olf)", ports="functional")
