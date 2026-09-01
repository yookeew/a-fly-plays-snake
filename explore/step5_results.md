# Step 5 — controls, first pass (2026-09-01)

**Setup.** No ES (the warm-start policy is a spike in parameter space that ES
falls off — see project notes). Each graph is scored by how well a linear
readout *regressed onto the food bearing* plays Snake — `warm_start_readout()` in
`train.py`. This isolates the one thing topology can affect here: how cleanly the
sensory signal reaches the descending (readout) neurons.

Board 12, 50 episodes/eval, 2 warm-start seeds averaged, `gain_init` swept over
{2,4,8,16} and reported at each arm's best. `inner_steps=8`.

## Numbers

| arm | food (mean of best-gain runs) | max | notes |
| --- | --- | --- | --- |
| random policy | 0.17 | — | floor |
| **real connectome** | **3.44** (0.42, 2.83, 4.42, 4.59, 4.93) | 15 | 5 port/readout draws |
| degree-preserving rewire | **5.10** (4.26, 5.09, 5.95) | 18 | 3 topology-scramble seeds |
| density-matched random | **5.59** (4.69, 5.36, 6.71) | 20 | 3 graph seeds |
| greedy_bot | 19.0 | — | shortsighted reference |

`compare.py` (real arm = draw 0 only) + `explore/real_arm_seeds.py` (draws 0–4).

## Read

- **The real connectome shows no advantage.** Its mean (3.4) is below both null
  arms (~5–5.6); its best draws (~4.5–4.9) only reach the *worst* draws of the
  nulls. Every arm clears the random-policy floor and loses badly to greedy.
- **The real connectome is less robust.** Rewire spans 4.3–6.0, random 4.7–6.7 —
  both tight. The real connectome spans 0.4–4.9 and the subgraph size itself
  swings (8.7k–12.6k) with the port/readout draw. The scrambles are *robustly
  mediocre*; the fly wiring is *inconsistent*.
- This is consistent with the *C. elegans* connectome-prior literature and with
  what CLAUDE.md predicted: on a task **outside the animal's behavioural
  domain**, the connectome's committed structure is not an asset. A null result
  here was always the likely one and is worth reporting.

## Do not overclaim

Fly-shaped sparse RNN, not a fly playing Snake. The trained parameters (readout,
and the gain we swept) are ours.

## Limitations — fix before this is more than suggestive

1. **Warm-start readout only, no ES.** The interesting question ("can the
   biophysical knobs be tuned to make the fly circuit good at this") is
   unanswered. Needs the readout-follows-dynamics reformulation.
2. **Low-competence regime.** All arms 3–6 food vs greedy's 19. We are comparing
   bad policies; the ordering could change with competent ones.
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
