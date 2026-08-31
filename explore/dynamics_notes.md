# Step 3d notes — dynamics sanity checks

Run: `.venv/Scripts/python.exe checks.py [flywire]`

All three checks pass on both connectomes. Numbers below are from the default
v1 subgraph (`load_flywire(n_obs=10)` → ~12.6k neurons, 64-neuron readout).

## 1. Reachability — OK

| graph | BFS hops port→readout | inner steps for >50% readout active |
| --- | --- | --- |
| synthetic | 2 | 3 |
| flywire | 3 | 5 |

`inner_steps=6` clears both, but flywire only by one step. **Bump `inner_steps`
to 8** for margin once training — it's cheap (one extra sparse matvec).

## 2. Stability — gain_init depends on the graph

`mean|h|` must settle into a band the readout can see (~0.05–3 × r_max, r_max=1).

| graph | healthy gains | pick |
| --- | --- | --- |
| synthetic | 1, 2, 4 | **gain_init = 2.0** |
| flywire | 4, 8, 16 | **gain_init = 8.0** |

Flywire needs ~4× more gain: it's sparser (mean in-degree 25 vs 100) and more
inhibitory (58% excitatory vs 70%), so `W @ r` cancels harder and signal decays
faster per hop. Pass these to `Brain.init_params(gain_init=...)` in step 4's ES.
ES tunes gain per-type from there.

## 3. Readout liveness — OK, with a caveat

The readout population is **centered** before the linear map (`model.py`) —
without it the single brightest descending neuron pins argmax and it never
flips, because input-driven logit variation (~0.01) is swamped by the random
per-neuron offset.

Even centered, ~1 in 3 random readout seeds still shows a stuck argmax at init.
That's fine — ES perturbs `b_out`/`W_out` and re-centers within a few
generations. **But**: if ES fitness is flat for the first ~20 generations,
suspect the readout init, not the reward. Bump `readout_scale` or reseed.

## Defaults going into step 4

```python
inner_steps = 8
gain_init   = 2.0   # synthetic / random-graph control
gain_init   = 8.0   # flywire
```
