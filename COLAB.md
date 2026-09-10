# Running the ES retina arms on Colab (T4)

The goal (option 1 in `explore/retina_results.md`): train the ~120 biophysical
knobs so the `retina` obs becomes usable, then ask whether the real FlyWire
subgraph beats its degree- and density-matched scrambles. A full-connectome CPU
generation is minutes; the T4 GPU forward pass (`model.Brain(device="cuda")`)
makes it tractable.

**Workflow:** the code lives in the repo. The notebook is a thin driver — clone,
install nothing (torch is preinstalled on Colab), launch `es_colab.py`. Every
generation checkpoints to Drive, so a reclaimed session resumes.

## One-time setup

1. Push the `es` branch: `git push origin es`.
2. Put the four CSVs — `connections_princeton.csv.gz`, `classification.csv.gz`,
   `column_assignment.csv.gz`, `consolidated_cell_types.csv.gz` — somewhere
   under `MyDrive/flybrain/` in the Drive web UI (~72 MB). Cell 1 finds them
   whether they sit in the folder root or a `data/` subfolder.
3. New Colab notebook, **Runtime → Change runtime type → T4 GPU**.

## Notebook cells

### Cell 1 — mount, clone, wire up data

```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/flybrain'      # your Drive folder

%cd /content
!rm -rf flybrain
!git clone https://github.com/yookeew/a-fly-plays-snake.git flybrain
%cd flybrain
!git checkout es

import glob, os, shutil
os.makedirs('data', exist_ok=True)
for s in set(glob.glob(f'{DRIVE}/**/*.csv.gz', recursive=True)):
    shutil.copy(s, 'data/')
os.makedirs(f'{DRIVE}/runs', exist_ok=True)
print('data/:', sorted(os.listdir('data')))

!nvidia-smi -L
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
```

`data/` must list at least those four files. Empty → `DRIVE` is wrong or the
upload didn't finish.

### Cell 2 — parity check (do this before trusting any run)

```python
!python explore/torch_parity.py
```

Expect `PARITY OK`. If the GPU path diverges, stop and fix `model._step_pop_torch`
before spending GPU hours.

### Cell 3 — benchmark: pick the substrate

Times two generations at `hops=3` and `hops=4`. `retina_results.md` already
showed `full=True` gives no R² gain over a subgraph, so only bench it (`--full`)
if you're curious.

```python
R = "/content/drive/MyDrive/flybrain/runs/_bench"
!python es_colab.py --arm real --obs retina --hops 3 --generations 2 --pop 48 --n-envs 6 --board 12 --out {R}
!python es_colab.py --arm real --obs retina --hops 4 --generations 2 --pop 48 --n-envs 6 --board 12 --out {R}
!rm -rf {R}
```

Read the `sec` on gen 2 (gen 1 includes the connectome load). Pick the deepest
`hops` whose per-generation time × 150 is a tolerable wall-clock (a T4 should do
`hops=3` in well under a minute/gen). Delete `runs/_bench/*` after.

### Cell 4 — synthetic sanity run (CLAUDE.md non-negotiable)

Confirms ES lifts a *random* graph above the floor on retina. If this fails, the
setup is broken — fix reward / curriculum / `--inner-steps` before the real arms.

Uses a board-size curriculum (`--board-min 6`, grows to 12): on a cold board-12
start every candidate eats ~0 and the ES gradient is noise, so start small where
random play hits food and there's variance to climb.

```python
# clear any checkpoint from an earlier (pre-curriculum) run first
!rm -f /content/drive/MyDrive/flybrain/runs/synthetic_retina_h3_*.pkl
!python es_colab.py --arm synthetic --obs retina --hops 3 --generations 120 --board-grow-every 8 --seed 0 --out /content/drive/MyDrive/flybrain/runs
```

What to watch:
- `bd` column ramps 6 → 12 (one step every 8 gens, full board by ~gen 50).
- `pop fit` should be clearly **positive and rising** on the small boards
  (gens 1–20). Flat noise there = still broken.
- `eval score mean` (always board 12) starts near 0 and should climb once `bd`
  passes ~9. Bar: past the printed random floor (~0.2), ideally 5+ by gen 120.

### Cell 5 — the three arms

Only once Cell 4 has cleared the bar. Run each in its own cell (or sequentially;
each ~1–3 h on a T4 at 150 gens). Same curriculum as Cell 4. `--resume` is on by
default, so re-running a cell after a disconnect continues.

```python
!python es_colab.py --arm real --obs retina --hops 3 --generations 150 --board-grow-every 8 --seed 0 --out /content/drive/MyDrive/flybrain/runs
```
```python
!python es_colab.py --arm rewire --obs retina --hops 3 --generations 150 --board-grow-every 8 --seed 0 --out /content/drive/MyDrive/flybrain/runs
```
```python
!python es_colab.py --arm synthetic --obs retina --hops 3 --generations 150 --board-grow-every 8 --seed 0 --out /content/drive/MyDrive/flybrain/runs
```

For CIs, repeat with `--seed 1 --seed 2` (each seed redraws ports/readout *and*
the scramble, so real vs control stays matched).

### Cell 5b — behaviour cloning (the pivot, PROJECT.md sec 4)

ES stalled (neither converges cold nor refines from the warm-start spike). BC
trains the same knobs by gradient descent against greedy_bot targets — dense
per-tick supervision, no reward/exploration/spike. Start with the sanity arm:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python bc.py --arm synthetic --obs retina --hops 3 --epochs 40 --out /content/drive/MyDrive/flybrain/runs
```

BC backprops once per game tick (h is detached between ticks), so memory is O(1)
in episode length. If you still OOM, drop `--batch` (default 24) or `--hops` /
`--max-ticks`.

Bar: `food` (honest, board 12) climbing past the ~0.2 floor as `loss` drops. If
a *random* graph's BC gets to 5+, the pipeline works — then the three arms:

```python
!python bc.py --arm real   --obs retina --hops 3 --epochs 60 --seed 0 --out /content/drive/MyDrive/flybrain/runs
!python bc.py --arm rewire  --obs retina --hops 3 --epochs 60 --seed 0 --out /content/drive/MyDrive/flybrain/runs
!python bc.py --arm synthetic --obs retina --hops 3 --epochs 60 --seed 0 --out /content/drive/MyDrive/flybrain/runs
```

North-star question: does `real` beat `rewire` / `synthetic`? Plus the
mechanistic check — re-run the bearing-R² probe on each trained `theta`.

### Cell 6 — compare

```python
import pickle, glob, numpy as np, matplotlib.pyplot as plt
runs = sorted(glob.glob('/content/drive/MyDrive/flybrain/runs/*_retina_h3_*.pkl'))
plt.figure(figsize=(8,5))
for f in runs:
    d = pickle.load(open(f,'rb'))
    ev = [(r['gen'], r['eval_score_mean']) for r in d['history'] if 'eval_score_mean' in r]
    g, s = zip(*ev)
    plt.plot(g, s, marker='o', label=f.split('/')[-1].replace('_retina_h3','').replace('.pkl',''))
    print(f"{f.split('/')[-1]:32s} last eval {s[-1]:.2f}  best {max(s):.2f}")
plt.xlabel('generation'); plt.ylabel('eval food (40 eps, board 12, reflex)')
plt.legend(); plt.grid(alpha=.3); plt.show()
```

The north-star question: does `real` end up **≥** `rewire` and `synthetic`? On
`feature` obs it did not (step 5). If it does here, retinotopic input + tuned
dynamics is a representation where the fly wiring is an asset.

### Cell 7 — mechanistic check (phase 5)

Did ES teach the *wiring* to compute pursuit, or just fit a reactive readout?
Re-run the bearing-R² probe (`explore/retina_deep.py` style) with the trained
`theta` in place of `init_params`, real vs scrambled. R² climbing from ~0 (see
`retina_results.md`) to something sizeable for `real` — and less for the
scrambles — is the wiring doing the work. *(Script TBD — flag when you get here.)*

## Notes / gotchas

- **`feature` baseline:** `--obs feature --feature-warm-start` reproduces step 5
  with the reflex + ES now bolted on — a useful control that the whole ES path
  works on the obs we know carries signal.
- **Colab limits:** free T4 sessions get reclaimed after ~12 h and on idle. Keep
  the tab active; `--resume` handles the rest. Checkpoints are ~10–50 KB.
- **`max_idle`** is set from board size in `es_colab.py`
  (`board²/2`) — long enough that a slow-but-working policy isn't starved.
- **Env stepping is CPU** even with the GPU forward pass — at `pop=48`
  (96 candidates) × `n_envs=6` that's 576 envs stepped in Python per tick. If a
  generation is dominated by that rather than the matvec, drop `n_envs` to 4.
