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
from model import Brain, N_ACT
from snake import SnakeEnv, FEATURE_NAMES, greedy_bot

FOOD_SIN = FEATURE_NAMES.index("food_sin")
FOOD_COS = FEATURE_NAMES.index("food_cos")
DANGER = {d: FEATURE_NAMES.index(f"danger_{d}")
          for d in ("L", "F", "R", "L2", "F2", "R2")}


# --------------------------------------------------------------------- rollout

def _food_dist(env):
    if env.food is None:
        return 0.0
    hx, hy = env.body[0]
    return abs(env.food[0] - hx) + abs(env.food[1] - hy)   # Manhattan


def rollout_pop(brain, thetas, n_envs, board, max_ticks, seed,
                obs_mode="feature", max_idle=200, shaping=0.0):
    """Episodes for a whole population at once. thetas (C, n_params).

    Returns (fitness, scores), each (C, n_envs). score = food eaten, ALWAYS the
    honest metric.

    fitness depends on `shaping`:
      shaping == 0  -> raw summed env reward (used for eval).
      shaping  > 0  -> a TRAINING reward built to make food-seeking the thing
                       that separates candidates:
                          +10  per food
                          + shaping * (dist_before - dist_after) toward food
                          -0.03 per step (so endless circling costs real fitness)
                       NO death penalty. Dying early is punished implicitly --
                       fewer steps means fewer chances at the +food and the
                       distance bonus. A -10 death spike, by contrast, makes
                       "approach the food then die" score worse than "circle
                       forever", which is the trap the earlier rewards fell in.
                       Potential-based: the distance term telescopes to
                       shaping*(d_start - d_end), so oscillating next to the
                       food nets zero.

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
                o, r_env, d = e.step(int(acts[c, i]))
                obs[c, i] = o
                ate = e.score > grew
                if shaping:
                    nd = _food_dist(e)
                    r = 10.0 * ate - 0.03
                    if not d and not ate:
                        r += shaping * (prev_d[c, i] - nd)
                    prev_d[c, i] = nd
                else:
                    r = r_env
                fitness[c, i] += r
                done[c, i] = d
        if done.all():
            break

    scores = np.array([[e.score for e in row] for row in envs], dtype=float)
    return fitness, scores


def evaluate(brain, theta, n_envs, board, max_ticks, seed, max_idle=200):
    """Honest score: raw reward (shaping=0), on whatever board is passed."""
    fit, sc = rollout_pop(brain, theta[None, :], n_envs, board, max_ticks, seed,
                          max_idle=max_idle)
    return fit.mean(), sc.mean(), sc.max()


def warm_start_readout(brain, theta, k=4.0, n_batches=60, seed=0):
    """Aim the readout at whatever descending neurons happen to carry the food
    bearing, so ES refines from a policy that already turns toward food instead
    of from the constant-action collapse it keeps falling into.

    This trains NOTHING in the connectome -- it is a smarter initialisation of
    the (already ours-to-train) linear readout. Drive the frozen brain with
    random observations whose food channels sweep all bearings, ridge-regress
    the descending activity onto [food_sin, food_cos], and wire that projection
    into turn-left / straight / turn-right. ES then takes over.
    """
    rng = np.random.default_rng(seed)
    params = brain.unpack_pop(theta[None, :])
    B = 64
    h = brain.initial_state_pop(1, B)
    R, Y = [], []
    for _ in range(n_batches):
        obs = rng.normal(0, 1, (1, B, brain.cx.n_obs)).astype(brain.dtype)
        ang = rng.uniform(-np.pi, np.pi, B)
        obs[0, :, FOOD_SIN] = np.sin(ang)
        obs[0, :, FOOD_COS] = np.cos(ang)
        h, _ = brain.step_pop(h, obs, params)
        r = np.clip(h[:, 0, :], 0.0, brain.r_max)          # (n, B)
        r_out = r[brain.out_idx]
        r_out = r_out - r_out.mean(axis=0, keepdims=True)   # match step_pop
        R.append(r_out.T)
        Y.append(np.c_[np.sin(ang), np.cos(ang)])
    R = np.concatenate(R)
    Y = np.concatenate(Y)
    A = np.linalg.solve(R.T @ R + 1e-2 * np.eye(R.shape[1]), R.T @ Y)  # (n_out,2)

    theta = theta.copy()
    t = brain._t
    W = np.zeros((N_ACT, brain.n_out))
    # sign fixed empirically: food_sin > 0 means food is to the RIGHT, so
    # turn-right wants +lateral. (ES would find this too, but starting on the
    # right side of it is the whole point.)
    W[0] = -k * A[:, 0]    # turn-left   <- food on the left
    W[2] = k * A[:, 0]     # turn-right  <- food on the right
    W[1] = k * A[:, 1]     # straight    <- food ahead
    theta[t:t + N_ACT * brain.n_out] = W.ravel()
    theta[t + N_ACT * brain.n_out:] = 0.0
    return theta


def warm_start_clone(brain, theta, k=6.0, n_batches=120, ridge=1e-2,
                     w_food=2.5, w_danger=1.5, w_far=0.5, seed=0):
    """Wire the readout to greedy_bot's decision rule, using regressed feature
    projections. The food-only warm start ignores the danger_* channels and so
    drives into its own tail; this one adds obstacle avoidance.

    Two parts, neither of which touches the connectome:

      1. Probe the frozen brain with random observations where BOTH the food
         bearing and all six danger_* channels are swept, and ridge-regress the
         descending activity onto those eight channels -> A (n_out x 8).
      2. Wire the three action logits to greedy_bot's actual logic -- go toward
         the food, minus the danger in that direction:

           left     =  -w_food*food_sin_proj - w_danger*dL_proj - w_far*dL2_proj
           straight =  +w_food*food_cos_proj - w_danger*dF_proj - w_far*dF2_proj
           right    =  +w_food*food_sin_proj - w_danger*dR_proj - w_far*dR2_proj

    Still an ES-free score of how cleanly each graph's wiring carries those
    features to the descending neurons; identical procedure on every step-5 arm.
    """
    rng = np.random.default_rng(seed)
    params = brain.unpack_pop(theta[None, :])
    B = 64
    h = brain.initial_state_pop(1, B)
    chans = [FOOD_SIN, FOOD_COS, DANGER["L"], DANGER["F"], DANGER["R"],
             DANGER["L2"], DANGER["F2"], DANGER["R2"]]
    R, Y = [], []
    for _ in range(n_batches):
        obs = rng.normal(0, 1, (1, B, brain.cx.n_obs)).astype(brain.dtype)
        ang = rng.uniform(-np.pi, np.pi, B)
        col = np.zeros((B, 8))
        col[:, 0], col[:, 1] = np.sin(ang), np.cos(ang)
        col[:, 2:] = rng.integers(0, 2, (B, 6))        # danger is 0/1 in the game
        obs[0, :, chans] = col.T.astype(brain.dtype)
        h, _ = brain.step_pop(h, obs, params)
        r = np.clip(h[:, 0, :], 0.0, brain.r_max)[brain.out_idx]
        R.append((r - r.mean(axis=0, keepdims=True)).T)
        Y.append(col)
    R = np.concatenate(R)
    Y = np.concatenate(Y)
    A = np.linalg.solve(R.T @ R + ridge * np.eye(R.shape[1]), R.T @ Y)  # (n_out, 8)
    fs, fc, dL, dF, dR, dL2, dF2, dR2 = A.T

    theta = theta.copy()
    t = brain._t
    W = np.zeros((N_ACT, brain.n_out))
    W[0] = w_food * (-fs) - w_danger * dL - w_far * dL2   # turn left
    W[1] = w_food * fc - w_danger * dF - w_far * dF2      # straight
    W[2] = w_food * fs - w_danger * dR - w_far * dR2      # turn right
    theta[t:t + N_ACT * brain.n_out] = (k * W).ravel()
    theta[t + N_ACT * brain.n_out:] = 0.0
    return theta


# ------------------------------------------------------------------- rank shape

def rank_normalise(x):
    """Map values to [-0.5, 0.5] by rank. Kills reward scale entirely -- only
    the ordering of candidates survives."""
    ranks = np.empty(len(x), dtype=float)
    ranks[np.argsort(x, kind="stable")] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


# -------------------------------------------------------------------------- ES

def es_train(brain, *, generations=150, pop=64, sigma=0.06, lr=0.03,
             n_envs=8, board=12, max_ticks=350, max_idle=90, shaping=0.3,
             gain_init=2.0, warm_start=True, readout_sigma_frac=0.25,
             seed=0, eval_every=10, out=None):
    rng = np.random.default_rng(seed)
    theta = brain.init_params(seed=seed, gain_init=gain_init)
    if warm_start:
        theta = warm_start_readout(brain, theta, seed=seed)
        fm, sm, sx = evaluate(brain, theta, 40, board, 3 * max_ticks,
                              seed=99, max_idle=200)
        print(f"warm-start readout: eval score mean {sm:.2f}  max {sx:.0f}",
              flush=True)
    d = theta.size

    # Per-block perturbation scale. The warm-started readout weights are large
    # and sensitive -- a full-size perturbation there collapses the policy to
    # constant-action and the gradient turns to noise. So perturb the readout
    # gently and let ES mostly work the biophysical knobs (gain/tau/bias), which
    # is the more faithful thing to be tuning anyway.
    pscale = np.ones(d)
    if warm_start:
        pscale[brain._t:] = readout_sigma_frac

    # Adam on the ES gradient estimate
    m = np.zeros(d)
    v = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8

    history = []
    t0 = time.time()
    for gen in range(1, generations + 1):
        E = rng.normal(size=(pop, d)) * pscale        # antithetic perturbations
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
            fm, sm, sx = evaluate(brain, theta, 40, board, 3 * max_ticks,
                                  seed=99, max_idle=200)
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
    ap.add_argument("--shaping", type=float, default=0.3,
                    help="potential-based food-distance bonus; 0 = raw reward")
    ap.add_argument("--neurons", type=int, default=6000,
                    help="synthetic only; smaller = faster milestone")
    ap.add_argument("--gain-init", type=float, default=None)
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--no-warm-start", action="store_true",
                    help="skip the food-bearing readout initialisation")
    ap.add_argument("--readout-sigma-frac", type=float, default=0.25,
                    help="perturbation scale for the readout block vs the "
                         "biophysical params; 0 = freeze the warm-started readout")
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
             warm_start=not a.no_warm_start,
             readout_sigma_frac=a.readout_sigma_frac, seed=a.seed, out=a.out)
