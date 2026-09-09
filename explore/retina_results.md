# Retina ports — the frozen-biophysics dead end (2026-09-08)

**Question.** `feature` obs hands the brain a pre-computed food bearing
(`food_sin`, `food_cos`). `retina` obs hands it a raw 7×7×3 egocentric patch and
the network has to *compute* the bearing. Can the FlyWire wiring do that
spatial→motor transform — and does the retinotopic port arrangement help?

**Method.** Held-out R² of a ridge fit from centred descending-neuron activity
onto `[food_sin, food_cos]`, probed by rolling `greedy_bot` on `retina` obs
through the frozen brain. This is exactly the signal `warm_start_rollout`
regresses; R² ≈ 0 means no linear readout can turn the retina frame into
steering, so the warm-start methodology cannot score retina at all.

Scripts: `explore/retina_iso.py` (quick, hops=2), `explore/retina_deep.py`
(full=True). Logs: `runs/retina_iso_quick.log`, `runs/retina_deep.log`.

## Numbers

| config | ro_reach | R²(sin) | R²(cos) |
| --- | --- | --- | --- |
| **feature / random ports** (reference) | 100% | **+0.63** | **+0.41** |
| retina / retinotopic — hops=2 | 56% | +0.21 | +0.09 |
| retina / shuffled — hops=2 | 56% | +0.09 | +0.06 |
| retina / random — hops=2 | 100% | +0.13 | +0.08 |
| retina / retinotopic — full, inner=16, gain=8 | 100% | +0.08 | +0.02 |
| retina / retinotopic — full, inner=28, gain=8 | 100% | +0.06 | +0.02 |
| retina / shuffled — full, inner=16 | 100% | −0.05 | +0.02 |
| retina / random — full, inner=16 | 100% | +0.10 | +0.08 |

(full retinotopic also run at gain=2 / inner {16,28}: R² ∈ [−0.03, +0.01].)

## Read

- **Retina through frozen biophysics carries no bearing.** Every retina config
  sits at the noise floor (|R²| ≲ 0.1) against feature's 0.63. A random-biophysics
  rate RNN does not propagate a lit pixel into a linearly-decodable heading at
  the descending neurons.
- **The retinotopic arrangement does nothing.** At full size, retinotopic
  (+0.08) ≈ shuffled (−0.05) ≈ random ports (+0.10). If anything random
  sensory ports edge it out. The hops=2 ordering (retinotopic 0.21 > shuffled
  0.09) was a small-subgraph artifact — ports sitting 2 hops from the readout
  leak a little signal directly; it vanishes once the signal has to cross the
  whole brain.
- **Depth was not the ceiling.** hops=2 → 56% readout reach was a real problem,
  but fixing it (full=True, 100% reach) did not move R². Neither did
  inner_steps 16→28 or gain 2→8.
- **Reflex does not rescue it** (`retina_iso_quick.log`: food 0.4 → 0.4 with the
  collision reflex, vs feature 3.8 → 16). The reflex protects a food-seeking
  signal; retina has none, so the snake just wanders.

## Consequence

The frozen-biophysics warm-start methodology (steps 4–5) **cannot evaluate
retina** — there is nothing at the readout to wire to. To ask "does the fly's
visual pursuit wiring help", the biophysical knobs (per-type gain / tau / bias,
~120 params) must first be tuned so the network computes *something*. That means
ES is a prerequisite for the retina arm, not an optional refinement.

Options:
1. **ES on biophysics + retina**, real vs scrambled. The honest continuation,
   but heavy — full-connectome forward pass is 66–110 ms/step, ES needs
   thousands of rollouts. A research project, not a cleanup.
2. **Stay on `feature` obs.** It works (R² 0.63, food ~20 with reflex) and the
   step-5 comparison — real connectome vs degree/density-matched scrambles — is
   the actual deliverable. Retina becomes a documented dead-end.
3. **Hybrid** (`ports="visual"`): `feature` obs, food channel routed to
   small-target LC neurons. Already tried — 4.7 food vs 8.7 for random ports
   (`runs/visual_ports.log`). Worse, not better.
