"""Is step 4 stuck on broken plumbing or a hard search?

Three probes on the synthetic graph, cheap:

  A. does the readout actually respond to the FOOD channels, or is it deaf to
     them?  (feed obs that differ only in food_sin/food_cos, watch the logits)
  B. random search: sample many parameter vectors, eval each. If the best of
     hundreds still eats ~0, the network cannot express food-seeking. If some
     eat 3-5, it is an optimisation problem, not a plumbing one.
  C. a hand-wired readout: connect food bearing -> turn directly through the
     readout bias/weights and see if the loop closes at all.
"""

import numpy as np

from brain import make_synthetic
from model import Brain, N_ACT
from snake import SnakeEnv, FEATURE_NAMES

N_OBS = len(FEATURE_NAMES)
FOOD_SIN, FOOD_COS = FEATURE_NAMES.index("food_sin"), FEATURE_NAMES.index("food_cos")

cx = make_synthetic(n_obs=N_OBS, n_neurons=4000, seed=0)
brain = Brain(cx, inner_steps=6)


def rollout1(theta, n=12, board=10, max_ticks=200, max_idle=70, seed=0):
    p = brain.unpack_pop(theta[None, :])
    tot = 0
    for k in range(n):
        e = SnakeEnv(board, board, max_idle=max_idle, seed=seed + k)
        o = e.reset()
        h = brain.initial_state_pop(1, 1)
        done = False
        while not done:
            h, a = brain.act_pop(h, o.reshape(1, 1, -1), p)
            o, _, done = e.step(int(a[0, 0]))
        tot += e.score
    return tot / n


# ---- A. readout sensitivity to the food channels -------------------------
print("A. readout response to food bearing")
theta0 = brain.init_params(seed=0, gain_init=2.0)
p = brain.unpack_pop(theta0[None, :])
rng = np.random.default_rng(0)
h = brain.initial_state_pop(1, 64)
base = rng.normal(0, 1, (1, 64, N_OBS)).astype(brain.dtype)
for _ in range(6):                       # settle
    h, _ = brain.step_pop(h, base, p)
# sweep food bearing across the 64 parallel copies, hold everything else
angles = np.linspace(-np.pi, np.pi, 64)
probe = base.copy()
probe[0, :, FOOD_SIN] = np.sin(angles)
probe[0, :, FOOD_COS] = np.cos(angles)
_, logits = brain.step_pop(h.copy(), probe, p)      # (1, N_ACT, 64)
lg = logits[0].T                                     # (64, N_ACT)
print(f"   logit range across food angle: "
      f"{np.array2string(lg.max(0) - lg.min(0), precision=4)}")
print(f"   argmax varies with food angle: "
      f"{len(np.unique(lg.argmax(1)))} distinct actions over the sweep")

# ---- B. random search ---------------------------------------------------
print("\nB. random search over parameters (60 draws x 12 episodes)")
best, best_theta = -1, None
for s in range(60):
    r = np.random.default_rng(1000 + s)
    theta = brain.init_params(seed=s, gain_init=r.uniform(1.5, 6.0),
                              readout_scale=r.uniform(0.2, 1.5))
    # also jitter the per-type gain/bias hard
    theta[:brain.n_types] += r.normal(0, 1.0, brain.n_types)             # alpha
    theta[brain.n_types:2 * brain.n_types] += r.normal(0, 0.8, brain.n_types)
    theta[2 * brain.n_types:3 * brain.n_types] += r.normal(0, 0.5, brain.n_types)
    sc = rollout1(theta, n=12, seed=0)
    if sc > best:
        best, best_theta = sc, theta
        print(f"   draw {s:3d}: mean food {sc:.2f}  <-- best")
print(f"   BEST random policy: {best:.2f} food/episode")

# ---- C. hand-wired food -> turn ---------------------------------------
print("\nC. can a hand-tuned readout close the loop?")
# brute a scalar that maps the food channels straight onto the L/R logits via
# whichever readout neurons happen to carry that signal
best_c = -1
for scale in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
    theta = brain.init_params(seed=0, gain_init=3.0, readout_scale=0.0)
    # readout weight block is the last N_ACT*n_out; set turn-left minus
    # turn-right to a random projection, scaled
    r = np.random.default_rng(7)
    t = brain._t
    w = r.normal(0, 1, (N_ACT, brain.n_out))
    theta[t:t + N_ACT * brain.n_out] = (scale * w).ravel()
    sc = rollout1(theta, n=12, seed=0)
    best_c = max(best_c, sc)
    print(f"   readout scale {scale}: {sc:.2f} food")
print(f"   best: {best_c:.2f}")
