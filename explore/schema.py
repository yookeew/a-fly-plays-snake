"""Step 2: learn the Codex schema. Not importable, not the loader -- just prints.

Run:  .venv/Scripts/python.exe explore/schema.py

Answers, with real numbers:
  1. how many neurons / edges survive, and what threshold Princeton already applied
  2. the literal strings in super_class / class / (cell) type
  3. how many descending neurons, and how to select them
  4. the nt_type distribution, and how many come back unknown
"""

import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)

DATA = "data"
CONN = f"{DATA}/connections_princeton.csv.gz"
CLASS = f"{DATA}/classification.csv.gz"
COLS = f"{DATA}/column_assignment.csv.gz"
VIS = f"{DATA}/visual_neuron_types.csv.gz"


def rule(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------------------
rule("FILE: connections_princeton.csv.gz  (the edge list)")
conn = pd.read_csv(CONN)
print("columns :", list(conn.columns))
print("rows    :", len(conn), " <- one row per (pre, post, neuropil)")
print(conn.head(3))
print("\nsyn_count describe:")
print(conn["syn_count"].describe())
print("min syn_count :", conn["syn_count"].min(),
      "  <- Princeton's pre-applied threshold (Codex 'Princeton' file is >=5)")
print("rows with syn_count < 5 :", (conn["syn_count"] < 5).sum())

pre = set(conn["pre_root_id"])
post = set(conn["post_root_id"])
print("\nunique pre_root_id  :", len(pre))
print("unique post_root_id :", len(post))
print("union (neurons in edge list) :", len(pre | post))

# collapse neuropil rows -> one edge per (pre, post)
pair = conn.groupby(["pre_root_id", "post_root_id"], sort=False)["syn_count"].sum()
print("distinct (pre,post) pairs after collapsing neuropil :", len(pair))

# ---------------------------------------------------------------------------
rule("FILE: classification.csv.gz  (per-neuron labels)")
cls = pd.read_csv(CLASS)
print("columns :", list(cls.columns))
print("rows    :", len(cls))
print(cls.head(3))
print("\nNOTE: no 'cell_type' column here. CLAUDE.md's 'cell_type' is stale.")
for col in ("flow", "super_class", "class"):
    rule(f"  classification['{col}']  value_counts (literal strings)")
    vc = cls[col].value_counts(dropna=False)
    print(vc)

# ---------------------------------------------------------------------------
rule("DESCENDING NEURONS: how many, how to select")
for col in ("flow", "super_class", "class"):
    hits = cls[cls[col].astype(str).str.contains("descend", case=False, na=False)]
    print(f"rows where {col} ~ 'descend' : {len(hits)}")
    if len(hits):
        print("   distinct values:", sorted(hits[col].unique()))
# also intrinsic / sensory sizes for context
for name, mask in {
    "super_class == 'sensory'": cls["super_class"] == "sensory",
    "super_class == 'visual_projection'": cls["super_class"] == "visual_projection",
    "super_class == 'central'": cls["super_class"] == "central",
}.items():
    print(f"{name:40s} : {mask.sum()}")

# ---------------------------------------------------------------------------
rule("nt_type distribution (from the edge list)")
print("per-EDGE nt_type value_counts:")
print(conn["nt_type"].value_counts(dropna=False))
print("\nedges with missing/unknown nt_type :",
      conn["nt_type"].isna().sum(), "NaN +",
      (conn["nt_type"].astype(str).str.lower().isin(["unknown", "nan", ""])).sum(),
      "explicit-unknown-ish")

# nt is really a property of the presynaptic neuron (Dale). Check consistency.
per_pre = conn.groupby("pre_root_id")["nt_type"].agg(lambda s: s.dropna().nunique())
print("\npre neurons with >1 distinct nt_type across their edges :",
      (per_pre > 1).sum(), "of", len(per_pre),
      "  <- want ~0 if we sign per presynaptic cell")

# dominant nt per pre neuron
dom = conn.dropna(subset=["nt_type"]).groupby("pre_root_id")["nt_type"].agg(
    lambda s: s.value_counts().idxmax())
rule("nt_type per PRESYNAPTIC neuron (dominant vote)")
print(dom.value_counts(dropna=False))
print("\nSIGN MAP: ACH->+1  GABA->-1  GLUT->-1 ; DA/OCT/SER->0 ; unknown-> ?")

# ---------------------------------------------------------------------------
rule("FILE: column_assignment.csv.gz  (retinotopic columns -- step 3 upgrade)")
col = pd.read_csv(COLS)
print("columns :", list(col.columns))
print("rows    :", len(col))
print(col.head(3))
print("distinct column_id :", col["column_id"].nunique())

rule("FILE: visual_neuron_types.csv.gz")
vis = pd.read_csv(VIS)
print("columns :", list(vis.columns))
print("rows    :", len(vis))
print(vis.head(3))
