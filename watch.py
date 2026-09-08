"""
Watch a connectome-driven brain play Snake, in the same window the human uses.

No training involved -- the connectome is frozen and the readout is warm-started
(`--warm clone`, the default: descending activity regressed onto the food
bearing + the six danger_* channels, wired to greedy_bot's rule; `--warm food`
is the older food-only version that ignores danger and drives into its tail).
This is the step-5 policy. Raw, it is not good -- ~4 food vs greedy's ~19,
because the readout drives the snake into its own body within ~20 ticks.

A downstream collision reflex (`lethal_actions`, on by default, `--no-reflex` to
drop it) masks any move that kills the snake this tick -- greedy_bot's veto
bolted onto the argmax. With it, board 12 / seed 0 goes ~4 -> ~20 food. The
long-horizon trapping deaths remain; killing those needs the connectome to
actually carry the danger signal, which is a later step.

    python watch.py                       # real FlyWire subgraph
    python watch.py --graph synthetic      # density-matched random graph
    python watch.py --graph rewire         # degree-preserving scramble of real
    python watch.py --seed 3 --gain 4      # different I/O draw / gain
    python watch.py --no-reflex               # raw readout, no collision veto
    python watch.py --headless --episodes 20   # just the scores, no window

Keys in the window: SPACE pause, R new game, ESC quit.
"""

import argparse

import numpy as np

from brain import make_synthetic, load_flywire, rewire_degree_preserving
from model import Brain
from train import warm_start_readout, warm_start_clone
from snake import play, FEATURE_NAMES, lethal_actions


class BrainPolicy:
    """Frozen connectome + warm-started readout, as a policy(env) -> action.

    Holds the rate-RNN hidden state across ticks (that is the whole point of a
    recurrent net) and clears it on .reset() between episodes.

    `reflex` bolts greedy_bot's refuse-to-die veto onto the argmax: any move that
    kills the snake this tick is masked out of the logits, as long as one option
    survives. The connectome routes the danger_* channels to the descending
    neurons too weakly to dodge its own body -- this is the cheap backstop until
    a later step wires danger into the logits directly.
    """

    def __init__(self, brain, theta, reflex=True):
        self.brain = brain
        self.params = brain.unpack_pop(theta[None, :])
        self.reflex = reflex
        self.reset()

    def reset(self):
        self.h = self.brain.initial_state_pop(1, 1)

    def __call__(self, env):
        o = env.observe().reshape(1, 1, -1).astype(self.brain.dtype)
        self.h, logits = self.brain.step_pop(self.h, o, self.params)
        logits = np.asarray(logits[0, :, 0], dtype=np.float64)
        if self.reflex:
            lethal = lethal_actions(env)
            if not lethal.all():                  # never mask the last way out
                logits[lethal] = -np.inf
        return int(np.argmax(logits))


def build(graph, seed, gain, inner, warm="clone", reflex=True):
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
    theta0 = brain.init_params(seed=0, gain_init=gain)
    ws = warm_start_clone if warm == "clone" else warm_start_readout
    theta = ws(brain, theta0, seed=0)
    print(f"{graph}: n={cx.n}, readout={len(cx.out_idx)}, gain_init={gain}, "
          f"warm-start={warm}, reflex={reflex}")
    return brain, BrainPolicy(brain, theta, reflex=reflex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="real",
                    choices=["real", "rewire", "synthetic"])
    ap.add_argument("--seed", type=int, default=0, help="port / readout draw")
    ap.add_argument("--warm", default="clone", choices=["clone", "food"],
                    help="clone = imitate greedy_bot; food = food-bearing only")
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--no-reflex", action="store_true",
                    help="disable the downstream collision reflex, to see how "
                         "often the raw readout drives into its own body")
    ap.add_argument("--inner-steps", type=int, default=8)
    ap.add_argument("--board", type=int, default=12)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--episodes", type=int, default=1)
    a = ap.parse_args()

    _, pol = build(a.graph, a.seed, a.gain, a.inner_steps, warm=a.warm,
                   reflex=not a.no_reflex)
    play(mode="brain", headless=a.headless, episodes=a.episodes, fps=a.fps,
         width=a.board, height=a.board, policy=pol)
