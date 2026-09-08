"""Isolate the steering-DN result: is it the *identity* of the turn-command DNs
that kills the readout, or just that there are only 24 of them?

func.log:
    random ports + random ro (64)   food 11.38
    functional ports + random (64)  food  7.38
    functional ports + steer (24)   food  2.15
    functional ports + locomotor(57) food  5.12
    random ports + steer ro (24)    food  1.18   <- steer is bad even off random ports

This script: random ports, RANDOM descending neurons, but n_readout swept
24 / 40 / 57 / 64 / 128. If food climbs smoothly with size and 24-random lands
near steer's 1-2, the steering DNs are not special -- a linear readout just
needs a wider basis. If 24-random still scores ~5+, the steering bottleneck is
doing real damage.
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

for nro in (24, 40, 57, 64, 128):
    cx = load_flywire(n_obs=N_OBS, ports="random", readout="random",
                      n_readout=nro, seed=0)
    reach = _r = None
    brain = Brain(cx, inner_steps=8)
    best = -1.0
    for g in GAINS:
        th = warm_start_rollout(brain, brain.init_params(seed=0, gain_init=g),
                                obs_mode="feature", seed=0)
        _, m, _mx = evaluate(brain, th, EPISODES, BOARD, 3 * BOARD * BOARD, seed=99)
        best = max(best, m)
    print(f"random ports + {nro:3d} random DNs   n={cx.n:6d}  best_food={best:5.2f}",
          flush=True)
