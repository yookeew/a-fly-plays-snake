# Step 5 — controls (2026-09-02, danger-aware readout)

**Setup.** No ES (the warm-start policy is a spike in parameter space that ES
falls off — see project notes). Each graph is scored by how well a linear
readout **wired to greedy_bot's rule** plays Snake: descending activity
regressed onto the food bearing *and* all six `danger_*` channels, combined as
"toward the food, minus danger in that direction" — `warm_start_clone()` in
`train.py`. This isolates the one thing topology can affect: how cleanly those
sensory features reach the descending (readout) neurons.

Board 12, 50 episodes/eval, 2 warm-start seeds averaged, `gain_init` swept over
{2,4,8,16} and reported at each arm's best. `inner_steps=8`.

## Numbers (compare.py, `runs/compare3.log`)

| arm | food (best-gain) | max | notes |
| --- | --- | --- | --- |
| random policy | 0.17 | — | floor |
| **real connectome** | **3.92** (4.61, 0.38, 2.42, 5.36, 6.85) | 20 | 5 port/readout draws |
| degree-preserving rewire | **7.42** (6.70, 7.16, 8.40) | 22 | 3 scramble seeds |
| density-matched random | **7.89** (6.66, 7.01, 10.01) | 26 | 3 graph seeds |
| greedy_bot | 19.0 | — | shortsighted reference |

Real arm over 5 draws (`explore/real_arm_seeds.py`, `runs/real_seeds3.log`):
mean 3.92, range **0.38–6.85**, subgraph size 8.7k–12.6k. The single best real
draw (6.85) just reaches the *worst* control run (6.66); the worst real draw
(0.38) barely plays. Controls are tight (6.7–10.0), real is not.

### Earlier, food-only readout (`warm_start_readout`, for the record)
real 3.44 (0.42–4.93, 5 draws) | rewire 5.10 | random 5.59. Same ordering, but
that readout ignores `danger_*` and dies 100% by self-collision — the
danger-aware one above is the better measure and widens the gap.

## Read

- **The real connectome is beaten by every control run.** Real mean 3.92 (best
  draw 6.85); the lowest of six control runs is 6.66. Both null means (~7.4,
  ~7.9) are ~2x real.
- **The real connectome is far less robust.** Real spans 0.38–6.85 across I/O
  draws and its subgraph size swings 8.7k–12.6k; the controls sit tight at
  6.7–10.0. Scrambles are *robustly mediocre*; the fly wiring is *erratic* —
  most draws bad, the occasional draw catches up.
- **More scrambling → better (and steadier) play.** real ~3.9 < degree-preserving
  rewire ~7.4 < fully random ~7.9. The fly's specific topology is a *liability*
  here; keeping its degree sequence recovers some of the loss but not all.
- Every arm clears the random-policy floor (0.17) and loses badly to greedy (19).
- Consistent with the *C. elegans* connectome-prior literature and with what
  CLAUDE.md predicted: on a task **outside the animal's behavioural domain**, the
  connectome's committed structure is not an asset. This null/negative result
  was always the likely one and is worth reporting.

## Do not overclaim

Fly-shaped sparse RNN, not a fly playing Snake. The trained parameters (readout,
and the gain we swept) are ours.

## Limitations — fix before this is more than suggestive

1. **Warm-start readout only, no ES.** The interesting question ("can the
   biophysical knobs be tuned to make the fly circuit good at this") is
   unanswered. Needs the readout-follows-dynamics reformulation.
2. **Low-competence regime.** All arms 4–10 food vs greedy's 19. We are comparing
   mediocre policies; the ordering could change with competent ones.
3. **Arbitrary I/O ports.** Real-arm variance is dominated by *which* sensory
   neurons are ports and *which* descendings are the readout. Retinotopic port
   mapping (Visual Neuron Columns) would remove much of this.
4. **Control seeds not matched to real seeds.** compare.py varied topology while
   holding draw-0's ports/readout; real_arm_seeds varied ports/readout. Redo
   with each control re-drawn per port/readout seed.
5. One board size, `feature` obs (not `retina`), one subgraph-extraction rule.

## Next

- Tighten the controls (limitation 4), add CIs, more seeds.
- Smaller/shallower flywire subgraph (`hops=1`) — warm-start may discriminate
  better when the signal has fewer hops to decay over.
- The retinotopic-scramble 4th arm (real connectome, medulla-column ports
  shuffled) — separates "connectome helps" from "spatial input helps". Needs
  `retina` obs.
- Eventually: the readout-follows-dynamics ES layer, then rerun all arms.
