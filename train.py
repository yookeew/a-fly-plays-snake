"""
Step 4: wire the brain to the game, train the ~300 knobs with evolution.

Do this FIRST, before touching the real connectome: run the whole pipeline on
the synthetic graph and confirm ES lifts the score above random. Even reaching 5
food is enough. "My plumbing is broken" and "the connectome does not help" look
identical from the outside; this separates them.

Rollout
-------
Grid Snake does not vectorise -- variable-length body, set membership, per-agent
food. Do not fight it. The network forward pass dominates by orders of magnitude
at ~12k neurons, so a plain Python loop over env objects is noise. W is shared, h
is stacked (batch, n), envs step in a for-loop, a done-mask freezes finished
agents while the batch runs on.

ES, not PPO
-----------
~300 parameters and a non-differentiable argmax readout. BPTT would hold
thousands of activation snapshots per episode; ES only ever calls the forward
pass. OpenAI-ES: antithetic sampling, rank-normalised fitness (immune to the
1000x gap between the step penalty and the food bonus), Adam on the estimated
gradient.

    python train.py --graph synthetic --generations 150
    python train.py --graph flywire   --generations 300 --gain-init 8
"""

import argparse
import pickle
import time

import numpy as np

from brain import make_synthetic, load_flywire
from model import Brain
from snake import SnakeEnv, FEATURE_NAMES, greedy_bot


# --------------------------------------------------------------------- rollout

def _food_dist(env):
    if env.food is None:
        return 0.0
    hx, hy = env.body[0]
    return abs(env.food[0] - hx) + abs(env.food[1] - hy)   # Manhattan


def rollout_pop(brain, thetas, n_envs, board, max_ticks, seed,
                obs_mode="feature", max_idle=200, shaping=0.0):
    """Episodes for a whole population at once. thetas (C, n_params).

    Returns (fitness, scores), each (C, n_envs). score = food eaten (the honest
    metric, always). fitness = summed env reward, plus -- if `shaping` > 0 --
    a potential-based bonus `shaping * (dist_before - dist_after)` for closing
    Manhattan distance to the food.

    Why shaping: with the raw +10-per-food / -0.01-per-step reward, a policy that
    never reaches food gets no gradient, and ES parks on the local optimum
    "circle until idle-death" (fitness ~ -5.5). The distance term is dense, and
    being potential-based it does not change which policy is optimal -- it just
    makes the road there visible. Training only; eval passes shaping=0.

    Every candidate sees the SAME n_envs boards (seed independent of candidate):
    common random numbers, on top of antithetic sampling.
    """
    C = thetas.shape[0]
    params = brain.unpack_pop(thetas)
    envs = [[SnakeEnv(board, board, obs=obs_mode, max_idle=max_idle,
                      seed=seed + 1000 * i)
             for i in range(n_envs)] for _ in range(C)]

    obs = np.zeros((C, n_envs, brain.cx.n_obs), dtype=brain.dtype)
    prev_d = np.zeros((C, n_envs))
    for c in range(C):
        for i in range(n_envs):
            obs[c, i] = envs[c][i].reset()
            prev_d[c, i] = _food_dist(envs[c][i])
    h = brain.initial_state_pop(C, n_envs)
    done = np.zeros((C, n_envs), dtype=bool)
    fitness = np.zeros((C, n_envs))

    for _ in range(max_ticks):
        h, acts = brain.act_pop(h, obs, params)          # (C, n_envs)
        for c in range(C):
            row = envs[c]
            for i in range(n_envs):
                if done[c, i]:
                    continue
                e = row[i]
                grew = e.score
                o, r, d = e.step(int(acts[c, i]))
                obs[c, i] = o
                if shaping and not d:
                    nd = _food_dist(e)
                    if e.score == grew:            # no food this tick
                        r += shaping * (prev_d[c, i] - nd)
                    prev_d[c, i] = nd
                fitness[c, i] += r
                done[c, i] = d
        if done.all():
            break

    scores = np.array([[e.score for e in row] for row in envs], dtype=float)
    return fitness, scores


def evaluate(brain, theta, n_envs, board, max_ticks, seed):
    fit, sc = rollout_pop(brain, theta[None, :], n_envs, board, max_ticks, seed)
    return fit.mean(), sc.mean(), sc.max()


# ------------------------------------------------------------------- rank shape

def rank_normalise(x):
    """Map values to [-0.5, 0.5] by rank. Kills reward scale entirely -- only
    the ordering of candidates survives."""
    ranks = np.empty(len(x), dtype=float)
    ranks[np.argsort(x, kind="stable")] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


# -------------------------------------------------------------------------- ES

def es_train(brain, *, generations=150, pop=64, sigma=0.06, lr=0.03,
             n_envs=8, board=12, max_ticks=350, max_idle=90, shaping=0.1,
             gain_init=2.0, seed=0, eval_every=10, out=None):
    rng = np.random.default_rng(seed)
    theta = brain.init_params(seed=seed, gain_init=gain_init)
    d = theta.size

    # Adam on the ES gradient estimate
    m = np.zeros(d)
    v = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8

    history = []
    t0 = time.time()
    for gen in range(1, generations + 1):
        E = rng.normal(size=(pop, d))                 # antithetic perturbations
        board_seed = int(rng.integers(1 << 30))       # CRN: shared this gen
        thetas = np.concatenate([theta + sigma * E, theta - sigma * E], axis=0)
        fit_grid, _ = rollout_pop(brain, thetas, n_envs, board, max_ticks,
                                  board_seed, max_idle=max_idle, shaping=shaping)
        fit = fit_grid.mean(axis=1)                   # (2*pop,)

        shaped = rank_normalise(fit)
        g = ((shaped[:pop] - shaped[pop:])[:, None] * E).sum(axis=0)
        g /= (2 * pop * sigma)                         # ascent direction

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** gen)
        vhat = v / (1 - b2 ** gen)
        theta += lr * mhat / (np.sqrt(vhat) + eps)

        row = {"gen": gen, "fit_mean": fit.mean(), "fit_best": fit.max(),
               "sec": time.time() - t0}
        if gen % eval_every == 0 or gen == 1:
            fm, sm, sx = evaluate(brain, theta, 40, 20, 900, seed=99)
            row.update(eval_fit=fm, eval_score_mean=sm, eval_score_max=sx)
            print(f"gen {gen:4d}  {row['sec']:6.0f}s  pop fit {fit.mean():+.2f}"
                  f"  |  eval score mean {sm:5.2f}  max {sx:3.0f}", flush=True)
        else:
            print(f"gen {gen:4d}  {row['sec']:6.0f}s  pop fit {fit.mean():+.2f}",
                  flush=True)
        history.append(row)

        if out:
            with open(out, "wb") as fh:
                pickle.dump({"theta": theta, "history": history,
                             "cfg": dict(pop=pop, sigma=sigma, lr=lr,
                                         board=board, gain_init=gain_init)}, fh)
    return theta, history


# ------------------------------------------------------------------------- main

def random_baseline(board=20, episodes=40, seed=123):
    rng = np.random.default_rng(seed)
    sc = []
    for ep in range(episodes):
        e = SnakeEnv(board, board, seed=seed + ep)
        done = False
        while not done:
            _, _, done = e.step(int(rng.integers(0, 3)))
        sc.append(e.score)
    return float(np.mean(sc)), int(np.max(sc))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="synthetic",
                    choices=["synthetic", "flywire"])
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--board", type=int, default=10)
    ap.add_argument("--max-ticks", type=int, default=220)
    ap.add_argument("--max-idle", type=int, default=90)
    ap.add_argument("--shaping", type=float, default=0.1,
                    help="potential-based food-distance bonus; 0 = raw reward")
    ap.add_argument("--neurons", type=int, default=6000,
                    help="synthetic only; smaller = faster milestone")
    ap.add_argument("--gain-init", type=float, default=None)
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    n_obs = len(FEATURE_NAMES)
    if a.graph == "flywire":
        cx = load_flywire(n_obs=n_obs)
        gain_init = a.gain_init or 8.0
        inner = a.inner_steps or 8
    else:
        cx = make_synthetic(n_obs=n_obs, n_neurons=a.neurons, seed=0)
        gain_init = a.gain_init or 2.0
        inner = a.inner_steps or 6      # synthetic reaches readout in ~3 hops

    brain = Brain(cx, inner_steps=inner)
    print(cx.summary(), flush=True)
    rb_mean, rb_max = random_baseline(board=20)
    print(f"\nrandom policy (20x20, 40 eps): mean {rb_mean:.2f}  max {rb_max}")
    print(f"greedy_bot target: mean 24.9  max 49\n", flush=True)

    es_train(brain, generations=a.generations, pop=a.pop, sigma=a.sigma,
             lr=a.lr, n_envs=a.n_envs, board=a.board, max_ticks=a.max_ticks,
             max_idle=a.max_idle, shaping=a.shaping, gain_init=gain_init,
             seed=a.seed, out=a.out)
