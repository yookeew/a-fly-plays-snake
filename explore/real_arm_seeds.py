"""The real arm of compare.py used ONE random port/readout draw. Is 2.83 that
draw's luck, or the connectome's? Run the real subgraph over several draws.

Each seed reselects the 200 port neurons and the 64 descending readout neurons
(load_flywire's `seed`), so this is the real-arm analogue of the rewire/random
seed sweeps in compare.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brain import load_flywire
from model import Brain
from train import warm_start_clone, evaluate
from snake import FEATURE_NAMES

N_OBS = len(FEATURE_NAMES)
BOARD, EPISODES, INNER = 12, 50, 8
GAIN_GRID = (2.0, 4.0, 8.0, 16.0)

for s in range(5):
    cx = load_flywire(n_obs=N_OBS, seed=s)
    brain = Brain(cx, inner_steps=INNER)
    best = (-1.0, 0.0, None)
    for gain in GAIN_GRID:
        runs = []
        for ws in (0, 1):
            theta = warm_start_clone(
                brain, brain.init_params(seed=0, gain_init=gain), seed=ws)
            _, m, mx = evaluate(brain, theta, EPISODES, BOARD,
                                3 * BOARD * BOARD, seed=99)
            runs.append((m, mx))
        runs = np.array(runs)
        if runs[:, 0].mean() > best[0]:
            best = (runs[:, 0].mean(), runs[:, 1].max(), gain)
    print(f"real subgraph seed {s}  n={cx.n:6d}  "
          f"best food {best[0]:5.2f} (max {best[1]:.0f})  @ gain {best[2]}",
          flush=True)
