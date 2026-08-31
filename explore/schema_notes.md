# Step 2 notes — Codex schema (FlyWire, downloaded 2026-08-31)

## Files on disk (`data/`)

| file | grain | columns (literal) |
| ---- | ----- | ----------------- |
| `connections_princeton.csv.gz` | one row per (pre, post, neuropil) | `pre_root_id, post_root_id, neuropil, syn_count, nt_type` |
| `classification.csv.gz` | one row per neuron | `root_id, flow, super_class, class, sub_class, hemilineage, side, nerve` |
| `column_assignment.csv.gz` | one row per optic neuron | `root_id, hemisphere, type, column_id, x, y, p, q` |
| `visual_neuron_types.csv.gz` | one row per optic neuron | `root_id, type, family, subsystem, category, side` |

## The four questions

**1. How many survive / what threshold is pre-applied?**
None. `syn_count` min is **1**; 1,588,394 rows (30%) are below 5. This is the
*no-threshold* Princeton export, not the default ≥5 one. So `min_syn` is **not
redundant** — apply ≥5 ourselves (standard FlyWire practice).
- raw: 5,342,446 rows → 3,732,460 directed edges (collapsed over neuropil), 138,584 neurons
- min_syn ≥ 5: 3,754,052 rows → **3,431,775 edges**, **138,533 neurons**
- `classification.csv.gz` has 139,255 neurons; every edge-list neuron is in it; 671 classified neurons have no edges.

**2. Literal strings.**
- `flow` (3): `intrinsic` 118464, `afferent` 19300, `efferent` 1491
- `super_class` (10): `optic` 77873, `central` 32381, `sensory` 16938, `visual_projection` 7684, `ascending` 1750, `descending` 1305, `sensory_ascending` 612, `visual_centrifugal` 522, `motor` 110, `endocrine` 80
- `class` (30 values **+ 31,664 NaN**): `optic_lobe_intrinsic` 77382, `visual` 11426, `Kenyon_Cell` 5177, `CX` 2878, `mechanosensory` 2674, `olfactory` 2281, `AN` 2276, `ALPN` 685, `LHLN` 517, `ALLN` 429, `gustatory` 408, `DAN` 331, `MBON` 96, `MBIN` 4, … (case varies: `Kenyon_Cell` vs `olfactory`)
- **There is no `cell_type` column.** CLAUDE.md's "group by `class` not `cell_type`" refers to a column that isn't in this release. Optic-cell fine types live in a *separate* file (`visual_neuron_types.type`, `column_assignment.type`, e.g. `T4b`, `Tm1`).
- Type-grouping key for ES params: use `class`, but it's NaN for 31k mostly-central neurons. Combined `super_class + "/" + class` gives **43 keys**, NaN-safe — good O(50) granularity.

**3. Descending neurons.**
`super_class == 'descending'` → exactly **1305**. All are `flow == 'efferent'`.
`side`: 650 right / 647 left / 8 center. No `class`-level or `flow`-level filter
needed; the `super_class` string is the selector. This is the readout population.

**4. nt_type distribution.**
Per edge: `ACH` 3,210,049 · `GABA` 1,172,932 · `GLUT` 826,380 · `DA` 63,704 ·
`SER` 40,396 · `OCT` 28,985. **Zero NaN, zero "unknown".**
NT is consistent per presynaptic neuron: **0 of 137,518** pre-neurons carry more
than one `nt_type` → signing is a plain per-neuron lookup, no voting needed.
Per presynaptic neuron: `ACH` 94,160 · `GLUT` 22,657 · `GABA` 18,374 · `SER`
1,435 · `DA` 786 · `OCT` 106.

Sign map (uppercase, note `GLUT` not `Glu`, `ACH` not `ACh`):
```
ACH  -> +1
GABA -> -1
GLUT -> -1        # GluCl usually inhibitory in fly
DA, SER, OCT -> 0 # modulatory; drop these edges in v1
```
Zeroing DA/SER/OCT removes ~133k edges and silences the output of ~2,327
neurons (they become input-only). Acceptable for v1; note it.

## Corrected column names for the loader

```python
EDGES   = ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]
NEURONS = ["root_id", "flow", "super_class", "class", "sub_class",
           "hemilineage", "side", "nerve"]
# join edges<->neurons on root_id (pre_root_id / post_root_id)
# nt_type is on EDGES; reduce to one value per pre_root_id for the sign vector
# type_id  = factorize(super_class + "/" + class.fillna("NA"))   # 43 groups
# out_idx  = where(super_class == "descending")                  # 1305
```

If a future re-download `KeyError`s here: `print(df.columns)` and update the two
lists above. Expected maintenance.
