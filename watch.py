"""
Watch a connectome-driven brain play Snake, in the same window the human uses.

No training involved -- the readout is warm-started (regressed onto the food
bearing, see train.warm_start_readout) and the connectome is frozen. This is the
step-5 policy, the one that scores ~3 food on the real graph and ~5 on the
controls. It is not good. That is the finding.

    python watch.py                       # real FlyWire subgraph
    python watch.py --graph synthetic      # density-matched random graph
    python watch.py --graph rewire         # degree-preserving scramble of real
    python watch.py --seed 3 --gain 4      # different I/O draw / gain
    python watch.py --headless --episodes 20   # just the scores, no window

Keys in the window: SPACE pause, R new game, ESC quit.
"""

import argparse

import numpy as np

from brain import make_synthetic, load_flywire, rewire_degree_preserving
from model import Brain
from train import warm_start_readout
from snake import play, FEATURE_NAMES


class BrainPolicy:
    """Frozen connectome + warm-started readout, as a policy(env) -> action.

    Holds the rate-RNN hidden state across ticks (that is the whole point of a
    recurrent net) and clears it on .reset() between episodes.
    """

    def __init__(self, brain, theta):
        self.brain = brain
        self.params = brain.unpack_pop(theta[None, :])
        self.reset()

    def reset(self):
        self.h = self.brain.initial_state_pop(1, 1)

    def __call__(self, env):
        o = env.observe().reshape(1, 1, -1).astype(self.brain.dtype)
        self.h, a = self.brain.act_pop(self.h, o, self.params)
        return int(a[0, 0])


def build(graph, seed, gain, inner):
    n_obs = len(FEATURE_NAMES)
    if graph == "synthetic":
        real = load_flywire(n_obs=n_obs, seed=seed)     # match its size
        cx = make_synthetic(n_obs, n_neurons=real.n, n_out=len(real.out_idx),
                            mean_in_degree=round(real.W.nnz / real.n),
                            frac_inhib=1 - float((real.W.data > 0).mean()),
                            n_types=real.n_types, seed=seed)
    elif graph == "rewire":
        cx = rewire_degree_preserving(load_flywire(n_obs=n_obs, seed=seed),
                                      seed=seed)
    else:
        cx = load_flywire(n_obs=n_obs, seed=seed)

    brain = Brain(cx, inner_steps=inner)
    theta = warm_start_readout(brain, brain.init_params(seed=0, gain_init=gain),
                               seed=0)
    print(f"{graph}: n={cx.n}, readout={len(cx.out_idx)}, gain_init={gain}")
    return brain, BrainPolicy(brain, theta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="real",
                    choices=["real", "rewire", "synthetic"])
    ap.add_argument("--seed", type=int, default=0, help="port / readout draw")
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--inner-steps", type=int, default=8)
    ap.add_argument("--board", type=int, default=12)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--episodes", type=int, default=1)
    a = ap.parse_args()

    _, pol = build(a.graph, a.seed, a.gain, a.inner_steps)
    play(mode="brain", headless=a.headless, episodes=a.episodes, fps=a.fps,
         width=a.board, height=a.board, policy=pol)
