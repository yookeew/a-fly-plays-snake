"""
Step 5: the controls. Without them the result means nothing.

The question is never "does the connectome network score X" -- it is "does the
connectome score better than a graph that kept its statistics but scrambled its
wiring". Three arms:

  1. real       -- the FlyWire subgraph, load_flywire().
  2. rewire     -- same in/out degree per node, same per-neuron weights and
                   signs, topology scrambled. rewire_degree_preserving().
  3. random     -- density-matched random graph, make_synthetic(), sized to the
                   real subgraph's n / mean-degree / readout / E-I ratio.

No ES here (see the project notes -- the warm-start policy is a spike ES cannot
climb). Each graph is scored by how well a READOUT WIRED TO GREEDY_BOT'S RULE
plays Snake -- descending activity regressed onto the food bearing AND the six
danger_* channels, then combined as "toward food, minus danger that way"
(train.warm_start_clone). The one thing topology can affect here: how cleanly
those sensory features reach the descending neurons.

gain_init is ours to set (a "put the dynamics in range" knob, not the fly's), so
each arm is swept over a small gain grid and reported at its OWN best -- every
arm gets its best shot, no cross-arm gain confound. rewire and random are run
over several seeds for a distribution.

Judge real vs the rewire / random arms, NOT vs greedy_bot. And the trained
parameters are ours -- this is a fly-shaped sparse RNN, not a fly playing Snake.

    python compare.py
"""

import numpy as np

from brain import make_synthetic, load_flywire, rewire_degree_preserving
from model import Brain
from train import warm_start_clone, evaluate, random_baseline
from snake import FEATURE_NAMES, SnakeEnv, greedy_bot

N_OBS = len(FEATURE_NAMES)
BOARD = 12
EPISODES = 50
WARM_SEEDS = (0, 1)
GAIN_GRID = (2.0, 4.0, 8.0, 16.0)
INNER = 8


def score_once(brain, gain, ws):
    theta = warm_start_clone(
        brain, brain.init_params(seed=0, gain_init=gain), seed=ws)
    _, mean_sc, max_sc = evaluate(brain, theta, EPISODES, BOARD,
                                  3 * BOARD * BOARD, seed=99)
    return mean_sc, max_sc


def score_arm(cx, name):
    brain = Brain(cx, inner_steps=INNER)
    best = (-1.0, 0.0, None)
    for gain in GAIN_GRID:
        runs = np.array([score_once(brain, gain, ws) for ws in WARM_SEEDS])
        m = runs[:, 0].mean()
        if m > best[0]:
            best = (m, runs[:, 1].max(), gain)
    print(f"  {name:34s}  n={cx.n:6d}  best food {best[0]:5.2f} "
          f"(max {best[1]:.0f})  @ gain {best[2]}", flush=True)
    return best


def greedy_ref(board, episodes=40, seed=7):
    sc = [SnakeEnv(board, board, seed=seed + e) for e in range(episodes)]
    out = []
    for e in sc:
        d = False
        while not d:
            _, _, d = e.step(greedy_bot(e))
        out.append(e.score)
    return float(np.mean(out)), int(np.max(out))


if __name__ == "__main__":
    real = load_flywire(n_obs=N_OBS)
    deg = round(real.W.nnz / real.n)
    einh = 1 - float((real.W.data > 0).mean())

    print(f"board {BOARD}   random {random_baseline(board=BOARD)[0]:.2f}   "
          f"greedy {greedy_ref(BOARD)[0]:.1f}\n", flush=True)

    print("REAL", flush=True)
    score_arm(real, "real connectome")

    print("\nREWIRE (degree-preserving, topology scrambled)", flush=True)
    for s in range(3):
        score_arm(rewire_degree_preserving(real, seed=s), f"rewire seed {s}")

    print("\nRANDOM (density-matched: n, degree, E/I, #types, #readout)", flush=True)
    for s in range(3):
        cx = make_synthetic(N_OBS, n_neurons=real.n, n_out=len(real.out_idx),
                            mean_in_degree=deg, frac_inhib=einh,
                            n_types=real.n_types, seed=s)
        score_arm(cx, f"random seed {s}")

    print("\njudge: real vs the rewire / random spreads, not vs greedy", flush=True)
