# flysnake

Can the wiring diagram of a fruit fly brain, used as a fixed network
architecture, learn to play an odor-guided Snake variant?

The FlyWire connectome (~139k neurons, ~50M synapses, CC-BY) supplies the
sparsity pattern and synaptic signs. Those are **never trained**. A small set of
biophysical parameters the EM cannot see — per-cell-type gain, time constant,
bias — plus a linear readout are fitted with evolution strategies.

## Status

| step | what | state |
| ---- | ---- | ----- |
| 1 | playable game + hand-coded baseline | **done** — `snake.py` |
| 2 | download Codex CSVs, learn the schema | next |
| 3 | CSVs → signed sparse matrix, dynamics sanity checks | |
| 4 | vectorise env, wire brain to game, ES training | |
| 5 | control arms (rewired null, random graph) | |

Baseline to beat: `greedy_bot` (move toward food, refuse to die this tick)
scores **mean 24.9, max 49** over 40 episodes on a 20×20 board with walls.
It is deliberately shortsighted and walls itself into pockets, so there is real
headroom for a policy with memory.

## Design decisions, and why

**The game is arcade Snake.** Grid, discrete steps, four-direction movement,
walls kill. Two deliberate deviations, and only two:

*Relative actions.* Turn left / straight / turn right, not four absolute
directions. Costs nothing — reversing is illegal in Snake anyway — and it is
what a descending-neuron readout naturally produces: three outputs, argmax.

*Egocentric observation.* The agent never receives a top-down board. A fly brain
has no machinery for allocentric maps; it has plenty for "what is in front of
me". Two observation modes: `feature` (10 hand-picked egocentric values) and
`retina` (7×7×3 patch rotated so the heading points up). **Use `retina` with the
connectome** — it is retinotopic, so it drops into a visual input population
instead of an abstract feature vector.

A continuous odor-plume variant lives in `snake_continuous.py`. It is a better
fit for fly biology and a worse fit for the actual goal, which is arcade Snake.
Kept for reference, not on the critical path.

**numpy, not torch, for v1.** ES only ever calls the forward pass, so autograd
buys nothing. The port to torch is mechanical and belongs to v2, where
mushroom-body plasticity does need gradients.

**ES, not PPO.** ~300 free parameters, and BPTT over 900 ticks × 6 inner steps
would mean holding 5400 activation snapshots. ES sidesteps that entirely.

**Signs come from neurotransmitter, per presynaptic neuron** (Dale's law).
ACh → +1, GABA and Glu → −1 (GluCl is usually inhibitory in fly). Dopamine,
octopamine and serotonin are modulatory; v1 zeroes them rather than pretending
they are fast synapses.

## Non-negotiables

- Never train the edge weights or flip a synaptic sign. That would discard the
  connectome, which is the entire point.
- Always run the control arms. A degree-preserving rewired connectome and a
  density-matched random graph. Without them you cannot distinguish "fly wiring
  did something" from "a sparse RNN with 2.7M edges did something" — and the
  *C. elegans* literature says connectome priors actively *hurt* on tasks
  outside the animal's behavioural domain.
- The trained parameters are ours, not the fly's. This is a fly-shaped sparse
  RNN, not a fly playing Snake. Do not overclaim in comments or writeups.

## Gotchas already paid for

- **Normalise incoming weights** (each neuron's total |w| → 1) before anything
  else. Skipping it makes activity explode on the first forward pass. This is
  the most common failure mode.
- **`inner_steps` must exceed the hop count.** ORN → descending neuron is
  roughly 4–6 synapses. Fewer inner steps per game tick and the input literally
  cannot reach the output.
- **Group cell types by `class`, not `cell_type`.** `cell_type` has ~8000
  values → ~24000 free parameters → ES crawls. `class` gives ~O(50).
- **Codex column names drift between releases.** A `KeyError` in the loader is
  expected maintenance. Print `df.columns` and adjust.
- **Trail buffer starts pre-filled** with copies of the spawn point. Without an
  `age < t` guard the snake collides with a tail it has not laid yet and dies on
  step 1 of every episode.
- **The retina rotation is easy to get backwards.** Local forward must map onto
  the heading, local right onto the heading turned clockwise. Invert it and the
  body renders sideways. Test: the patch must be *identical* for all four
  headings given the same relative layout.
- **The tail cell vacates each tick**, so moving onto it is legal unless the
  snake is growing. Miss this and the snake dies chasing its own tail end.
- **Start with a subgraph**, ~10–20k neurons within a few hops of the input
  ports, not all 139k. It loads fast and you can actually see where the dynamics
  break.

## Stack

Python 3.13 or 3.14 in a venv. `numpy`, `pygame-ce`, `pandas`, `scipy`,
`matplotlib`. Use **pygame-ce**, not `pygame` — upstream has no wheels for 3.14
and its build config still imports the removed `distutils.msvccompiler`. Never
install both; they collide on the `pygame` module name.

Connectome CSVs go in `data/` and are gitignored. Get them from
codex.flywire.ai → Info → Download Data.

## Running

```bash
python snake.py --mode human                        # arrows/WASD, A/D to turn
python snake.py --mode human --obs retina           # see the egocentric patch
python snake.py --mode bot                          # greedy baseline
python snake.py --mode bot --headless --episodes 40 # score it
```

## Conventions

- Actions are `0 = turn left, 1 = straight, 2 = turn right`. Fixed.
- Observation order is load-bearing; the brain injects into input populations in
  exactly the order `FEATURE_NAMES` lists, or raveled `(3, K, K)` for retina.
- Connectome producers (`make_synthetic`, `load_flywire`) return the same
  `Connectome` dataclass. Build against synthetic first; it doubles as a control.
- `W[i, j]` is the weight **from j into i**.
- Comments explain *why*, especially where a choice looks arbitrary or where a
  previous version was wrong. Do not strip those.

## Later (v2, not now)

Reward enters at the biologically correct port instead of via the optimizer:
eating fires sugar gustatory receptor neurons → PAM dopaminergic cluster →
plasticity at Kenyon-cell→MBON synapses → MBON output biases turning. Learns
worse than ES, but "does the fly's own credit-assignment circuit work when you
run it forward" is the interesting question, and nobody has closed that loop on
a whole-brain connectome. Also deferred: FlyGym/NeuroMechFly for a real body.