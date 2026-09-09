# flysnake — project journal

A running narrative of what this project is, what's in it, and the path it has
taken. `CLAUDE.md` holds the design decisions and non-negotiables; this file
holds the *story*, including the dead ends, so the next person (or the next
session) doesn't re-walk them.

---

## 1. What this is

**Surface question.** Can the wiring diagram of a fruit-fly brain — the FlyWire
connectome, ~139k neurons, ~2.7M signed synapses — used as a *fixed* network
architecture, learn to play an egocentric Snake?

**The connectome is frozen.** Edge weights and synaptic signs are never trained.
What gets fitted is a small set of biophysical knobs the electron microscope
can't see — per-cell-type gain, time constant, bias (~120 numbers) — plus a
linear readout off the descending neurons (~190 numbers). ~300 parameters
total. This is a *fly-shaped sparse RNN*, not a fly playing Snake.

**The real question (north star).** flysnake is a testbed. The point is:

> How do you translate a task so it lands inside an animal brain's actual
> behavioural domain — so the connectome's committed structure becomes an
> **asset** rather than a liability?

The result to characterise / overturn is the *C. elegans* connectome-prior
finding: connectome priors **help** on tasks inside the animal's repertoire and
**hurt** outside it. flysnake has so far reproduced the "hurt" half. The
interesting direction is the levers that move a task "inside": input encoding,
which neurons are I/O ports, what the readout asks for, task timescale.

---

## 2. What's in the repo

### The game and the model
| file | what |
| ---- | ---- |
| `snake.py` | Arcade Snake. Grid, discrete steps, walls kill. Two deviations: **relative** actions (turn L / straight / turn R) and **egocentric** observation — `feature` (10 hand-picked values incl. food bearing + `danger_*`) or `retina` (7×7×3 patch, heading-up). Also `greedy_bot` (the baseline: move toward food, refuse to die this tick — scores ~20) and `lethal_actions` (greedy's veto, exposed as a reusable mask). |
| `brain.py` | Connectome → signed sparse matrix. `Connectome` dataclass; `load_flywire()` (real subgraph, several port/readout schemes); `make_synthetic()` and `rewire_degree_preserving()` (the control arms); `normalise_incoming()`; `_reach()`. |
| `model.py` | `Brain` — a rate RNN over a frozen `W`. `h = (1-α)h + α(gain·(W@r) + bias + I)`, iterated `inner_steps` per game tick. numpy by default; `device="cuda"` moves the batched population forward pass onto torch sparse (for GPU ES). |
| `checks.py` | Reachability / stability / readout-liveness sanity checks, run before trusting any dynamics. |

### Training and evaluation
| file | what |
| ---- | ---- |
| `train.py` | `rollout_pop` (vectorised episodes over a candidate population), `evaluate` (honest food score), the warm starts (`warm_start_readout`, `warm_start_clone`, `warm_start_rollout` — regress descending activity onto env features, wire the readout to greedy's rule), and `es_train` (OpenAI-ES: antithetic, rank-normalised, Adam, board-size curriculum). |
| `compare.py` | Step 5. The 3-arm control comparison (real / rewire / synthetic), scored by warm-start performance. |
| `watch.py` | Watch a connectome-driven policy play in the pygame window. Frozen connectome + warm-started readout + the collision reflex. |
| `es_colab.py` | ES entry point for a GPU run — one arm per invocation, Drive checkpoint, resume. |
| `COLAB.md` | The Colab notebook recipe for the ES run. |

### `explore/` — scratch investigations, each with a `_notes.md` or `_results.md`
`schema*` (connectome file schema), `dynamics_notes.md` (gain/tau tuning),
`step5_results.md` (the control result), `retina_iso.py` / `retina_deep.py` /
`retina_results.md` (the retina investigation), `readout_size.py`,
`visual_ports.py`, `probe_policy.py`, `torch_parity.py`.

---

## 3. The arc

### Steps 1–3 — game, connectome, model
Built `snake.py`, the FlyWire loader, the rate RNN, and the dynamics checks.
Tuned `inner_steps=8`, `gain_init` ~2 (synthetic) / ~8 (flywire). Key gotcha
paid for: **normalise incoming weights** (total |w| → 1 per neuron) or activity
explodes on the first forward pass.

### Step 4 — wire brain to game, train with ES
Batched OpenAI-ES works mechanically. But it **cannot improve the warm-start
policy** — that policy is a sharp spike in parameter space (the readout is
regressed to descending activity at one exact biophysics config; any
perturbation falls off it). So the project pivoted: **score each graph by its
ES-free warm-start performance**, isolating the one thing topology can affect —
how cleanly the sensory features reach the descending neurons.

### Step 5 — the controls, and the result

`feature` obs, board 12, danger-aware warm-start readout, no ES:

| arm | food (best gain) | notes |
| --- | --- | --- |
| random policy | 0.17 | floor |
| **real connectome** | **3.9** (0.4–6.9 over 5 I/O draws) | erratic |
| degree-preserving rewire | **7.4** | tight |
| density-matched random | **7.9** | tight |
| greedy_bot | 19–25 | reference |

**The real connectome is beaten by every control run, and is far less robust.**
More scrambling → better and steadier play. A clean reproduction of the
*C. elegans* "connectome prior hurts outside the behavioural domain" finding.
(Do not overclaim: fly-shaped sparse RNN, trained params ours. Writeup:
`explore/step5_results.md`.)

### The collision reflex (2026-09-08)

The raw warm-start readout drives the snake into its own body within ~20 ticks
(the RNN has hysteresis — a turn continues into a spiral; the `danger_*`
channels reach the descending neurons too weakly and too late). Added
`snake.lethal_actions` — greedy_bot's refuse-to-die veto, as a mask applied to
the logits before argmax. A "brainstem" reflex, defensible (flies have fast
descending collision reflexes).

Effect on the real connectome + warm start, board 12: **4 → 20 food**. It lifts
every arm about equally (rewire/synthetic → ~25), so the step-5 ordering
survives — arguably cleaner, since it's no longer confounded by everything
dying instantly. Deaths become genuine long-horizon self-trapping.

### The retina investigation — a dead end through frozen biophysics

`feature` obs hands the network a *pre-computed* food bearing. `retina` obs
hands it a raw patch and the network must *compute* the bearing. Question: can
the FlyWire wiring do that spatial→motor transform, and does the retinotopic
port arrangement help?

Diagnostic: held-out R² of a linear fit from descending-neuron activity onto
`[food_sin, food_cos]`, probed by rolling greedy_bot. That's exactly what the
warm start regresses; R² ≈ 0 means no readout wiring can rescue retina.

| config | R²(sin) | R²(cos) |
| --- | --- | --- |
| **feature / random ports** (reference) | +0.63 | +0.41 |
| retina / retinotopic — full connectome | +0.08 | +0.02 |
| retina / shuffled ports — full | −0.05 | +0.02 |
| retina / random ports — full | +0.10 | +0.08 |

**Retina through frozen biophysics carries no bearing.** Every config sits at
the noise floor. The retinotopic *arrangement* does nothing — retinotopic ≈
shuffled ≈ random ports. Depth wasn't the ceiling (fixing port→readout
reachability from 56% to 100% didn't move it); neither did more inner steps or
higher gain. The reflex doesn't rescue it (food 0.4 → 0.4 — no steering signal
to protect). A random-biophysics rate RNN just doesn't turn a lit pixel into a
decodable heading. Writeup: `explore/retina_results.md`.

**Consequence:** the frozen-biophysics warm-start methodology *cannot evaluate
retina*. To ask "does the fly's visual wiring help", the biophysical knobs must
first be *trained* so the network computes something.

### The ES attempt — and why it stalled

Plan (option 1): train the ~120 biophysical knobs with ES so retina becomes
usable, then run real vs scrambled. A full-connectome CPU generation is
minutes, so this needed GPU.

Built: `model.Brain(device="cuda")` (torch-sparse GPU forward pass, numpy
everywhere else), `es_train` with `obs_mode`/`reflex` threading + checkpoint
resume + a board-size curriculum, `es_colab.py`, `explore/torch_parity.py`,
`COLAB.md`. Parity CPU-vs-GPU: OK.

Then it didn't work:

- **Synthetic + retina, cold start:** 33 generations, `eval` frozen at 0.47
  (the floor). `pop fit` looked positive but that was small-board + shaping
  inflating the baseline; it never trended up. The board-12 eval trajectory was
  *bit-identical* across generations — vision wasn't influencing the action at
  all.
- **Synthetic + feature, cold start** (feature obs *does* carry the bearing,
  R²=0.63): 40 generations, `eval` still frozen at 0.47.
- **Synthetic + feature + warm start:** the warm start gives ~22 food; ES's
  **first step collapses it to 0**. The step-4 "spike" failure, still present
  *with* the reflex.

Likely cause: `readout_sigma_frac` (meant to protect the fragile warm-started
readout by perturbing it gently) only scales the exploration *noise* — then
Adam's per-coordinate normalisation (`mhat/√vhat ≈ sign(g)`) throws the
magnitude away and steps every parameter by ~`lr` regardless. The readout gets
a full-size kick every generation. `readout_sigma_frac` is effectively a no-op
with Adam. This may be why ES has never worked in this project.

**Verdict:** ES here neither converges from a cold start nor refines from a warm
one. Fixable in principle (CMA-ES; SNES; scale the *update* by the block mask;
drop Adam) but it's an optimiser project of its own, and the free Colab GPU
quota is a real constraint on iterating.

---

## 4. The pivot — behaviour cloning

The forward pass **is** differentiable with respect to the biophysical knobs and
the readout — only the argmax action selection isn't. So instead of a
gradient-free reward search, use dense supervised targets:

1. Roll out `greedy_bot` (plays at ~20 food). Collect `(retina_obs,
   greedy_action)` pairs.
2. Train the biophysics + readout by **gradient descent** — cross-entropy
   between the network's softmax-over-3-actions and greedy's action.
3. Backprop through *one game tick* (the `inner_steps` unroll, 8–16 steps),
   detaching `h` between ticks. The "5400-snapshot BPTT" objection that ruled
   out gradients in v1 doesn't apply to truncated single-tick BC.

Why this should work where ES didn't:

- **Attacks R² ≈ 0 directly.** The warm start only fits a *linear* readout on
  frozen activity. BC also tunes the biophysics, so the network's internal
  representation *becomes* linearly decodable — the exact thing that's missing.
- **No reward, no exploration, no spike.** Dense per-timestep targets; the loss
  surface is ordinary supervised-learning-shaped.
- **Capacity option:** with gradients, per-*neuron* gain/tau/bias (~3n knobs) is
  cheap, where per-type (~120) may be too few to carve a pursuit path out of
  arbitrary wiring. Same "EM can't see it" justification.

Still within the non-negotiables: `W` and the signs are untouched; the trained
parameters are still ours, not the fly's.

**Then the controls, unchanged in spirit:** BC-train real / rewire / synthetic
under an identical budget, compare honest food score, and — the mechanistic
check — re-measure the bearing R² on each *trained* network. If real ends up
above the scrambles *and* its wiring learned to carry the bearing where the
scrambles couldn't, that's a task representation where the connectome is an
asset — the north-star result.

Fallbacks if BC also can't lift real above the scrambles: feed **Δretina**
(motion, which fly visual neurons actually encode), a coarser retinotopic code,
a hybrid feature+retina obs, or pivot the task itself to odor-plume tracking
(`snake_continuous.py`), which is squarely in the fly's behavioural domain.

---

## 5. Status

- **Branch `es`** holds the reflex, the retina probes, and the (stalled) ES/GPU
  scaffolding.
- **Deliverable in hand:** the step-5 result — real connectome loses to
  degree/density-matched scrambles on `feature`-obs Snake, reflex or not.
- **Now building:** behaviour cloning (`train.py` / a new `bc.py`), torch,
  greedy_bot targets, per-neuron biophysics option.
- **Open controls debt** (from `step5_results.md`): match control seeds to real
  I/O seeds, add CIs, more seeds.
