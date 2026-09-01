"""
Step 3: connectome -> signed sparse matrix -> a rate RNN with stable dynamics.

No game attached here. `snake.py` never imports this and this never imports
`snake.py`. Step 4 is what wires them together.

Two producers sit behind one `Connectome` dataclass:

  * `make_synthetic()`  -- a random sparse graph, sensory -> inter -> output.
    Not a throwaway. This is the DENSITY-MATCHED CONTROL ARM from step 5: if the
    real connectome cannot beat this, the fly wiring did nothing.
  * `load_flywire()`    -- the real thing. Step 3b. Not in this file yet.

Convention, everywhere: `W[i, j]` is the weight FROM j INTO i. So row i is
everything feeding neuron i, and `W @ r` is the input each neuron receives.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# Neurotransmitter -> synaptic sign, applied per PRESYNAPTIC neuron (Dale's law:
# the sign is a property of the cell, not the individual synapse). The modulatory
# transmitters are zeroed in v1 rather than pretending they are fast synapses --
# see CLAUDE.md. Keys are the literal uppercase strings from the Codex edge file.
NT_SIGN = {"ACH": +1.0, "GABA": -1.0, "GLUT": -1.0,
           "DA": 0.0, "SER": 0.0, "OCT": 0.0}


@dataclass
class Connectome:
    """Everything the model needs and nothing it does not.

    W        (n, n) signed, incoming-normalised CSR. W[i, j] = j -> i.
    type_id  (n,)   int in [0, n_types). Groups neurons for per-type gain / tau /
             bias. Grouped by `class`, never `cell_type` -- see CLAUDE.md.
    ports    list of length n_obs. ports[c] is an int array of neuron indices
             that observation channel c injects current into. Order is
             load-bearing: it must match FEATURE_NAMES / the raveled retina.
    out_idx  (n_out,) int. The readout population -- descending neurons for the
             real connectome. The linear map off these to 3 logits is a TRAINED
             parameter, so it does not live here.
    """

    W: sp.csr_matrix
    type_id: np.ndarray
    ports: list
    out_idx: np.ndarray

    @property
    def n(self):
        return self.W.shape[0]

    @property
    def n_types(self):
        return int(self.type_id.max()) + 1

    @property
    def n_obs(self):
        return len(self.ports)

    def summary(self):
        W = self.W
        deg_in = np.diff(W.indptr)
        absW = np.abs(W.data)
        row_norm = np.abs(W).sum(axis=1).A1
        live = row_norm > 1e-9
        return (
            f"Connectome  n={self.n}  edges={W.nnz}  "
            f"density={W.nnz / self.n**2:.2e}\n"
            f"  in-degree   mean {deg_in.mean():.1f}  max {deg_in.max()}  "
            f"zero-input neurons {int((deg_in == 0).sum())}\n"
            f"  |w|         mean {absW.mean():.3f}  max {absW.max():.3f}\n"
            f"  incoming-normalised: {live.sum()}/{self.n} rows sum to 1 "
            f"(rest have no input)  err {np.abs(row_norm[live] - 1).max():.2e}\n"
            f"  excitatory edges {np.mean(W.data > 0):.2%}  "
            f"types {self.n_types}  ports {self.n_obs}  readout {len(self.out_idx)}"
        )


# --------------------------------------------------------------------- helpers

def normalise_incoming(W):
    """Scale each row so its total |weight| is 1. Rows with no input stay zero.

    This is THE most common failure mode for these models: skip it and activity
    explodes on the very first forward pass, because a neuron with 200 unit-weight
    inputs sees an input 200x too large. Do it once, here, before anything else.
    """
    W = sp.csr_matrix(W, dtype=np.float64)
    row_abs = np.abs(W).sum(axis=1).A1
    scale = np.zeros_like(row_abs)
    np.divide(1.0, row_abs, out=scale, where=row_abs > 1e-12)
    return sp.diags(scale) @ W


def _dedupe_edges(pre, post, n):
    """Collapse duplicate (pre, post) pairs, drop self-loops. Returns unique
    pre, post arrays."""
    keep = pre != post
    pre, post = pre[keep], post[keep]
    key = pre.astype(np.int64) * n + post.astype(np.int64)
    key = np.unique(key)
    return (key // n).astype(np.int32), (key % n).astype(np.int32)


# ---------------------------------------------------------------- synthetic arm

def make_synthetic(n_obs, n_neurons=12_000, n_out=64, mean_in_degree=22,
                   frac_sensory=0.05, frac_inhib=0.30, n_types=40,
                   weight_spread=0.4, seed=0):
    """Random sparse graph with a soft sensory -> inter -> output gradient.

    Matched to the real subgraph on: node count, mean in-degree (~22 for the
    default flywire subgraph), E/I ratio, Dale's law (sign per presynaptic
    neuron), incoming normalisation. NOT matched on degree *distribution* or any
    topology -- that is what the degree-preserving rewire control (step 5, arm 2)
    is for. This is arm 3, the loosest null.
    """
    rng = np.random.default_rng(seed)

    n_sensory = max(n_obs, int(round(n_neurons * frac_sensory)))
    assert n_sensory + n_out < n_neurons, "no room left for interneurons"
    sens_idx = np.arange(n_sensory)
    out_idx = np.arange(n_neurons - n_out, n_neurons)

    # coarse stage per neuron: 0 sensory, 1 interneuron, 2 output
    stage = np.ones(n_neurons, dtype=np.int8)
    stage[sens_idx] = 0
    stage[out_idx] = 2

    # Acceptance probability by (pre stage, post stage). Forward is favoured,
    # feedback is allowed but rare -- enough recurrence for the network to hold
    # memory, not so much it is a soup. Rows: pre stage. Cols: post stage.
    P = np.array([[0.20, 1.00, 0.10],     # sensory -> {sensory, inter, out}
                  [0.05, 0.50, 1.00],     # inter   -> ...
                  [0.02, 0.25, 0.20]])    # output  -> ... (a little feedback)

    # oversample candidate edges, thin by the acceptance matrix, then subsample
    # to hit mean_in_degree exactly -- the acceptance step's yield is hard to
    # predict (it depends on how pre/post stages line up) so we overshoot and cut.
    target_edges = n_neurons * mean_in_degree
    n_cand = int(target_edges / P.mean() * 3)
    pre = rng.integers(0, n_neurons, size=n_cand, dtype=np.int32)
    post = rng.integers(0, n_neurons, size=n_cand, dtype=np.int32)
    accept = rng.random(n_cand) < P[stage[pre], stage[post]]
    pre, post = _dedupe_edges(pre[accept], post[accept], n_neurons)
    if pre.shape[0] > target_edges:
        pick = rng.choice(pre.shape[0], size=target_edges, replace=False)
        pre, post = pre[pick], post[pick]

    # Dale: each neuron is excitatory or inhibitory; every edge it sends carries
    # that sign. frac_inhib=0.30 matches the real connectome, where GABA + GLUT
    # presynaptic neurons are ~30% of the total (18374 + 22657 of 135766).
    sign = np.where(rng.random(n_neurons) < frac_inhib, -1.0, 1.0)

    mag = np.exp(rng.normal(0.0, weight_spread, size=pre.shape[0]))
    data = mag * sign[pre]

    W = sp.coo_matrix((data, (post, pre)),               # post = i, pre = j
                      shape=(n_neurons, n_neurons)).tocsr()
    W.sum_duplicates()
    W = normalise_incoming(W)

    type_id = rng.integers(0, n_types, size=n_neurons).astype(np.int32)

    # ports: hand each observation channel a disjoint slab of sensory neurons
    ports = [np.asarray(a, dtype=np.int32)
             for a in np.array_split(sens_idx, n_obs)]

    return Connectome(W=W, type_id=type_id, ports=ports, out_idx=out_idx)


def _reach(W, seeds, hops, backward=False):
    """Boolean mask of neurons reachable from `seeds` in <= `hops` steps.

    Sign-blind, on the adjacency of W. `backward=True` walks edges in reverse
    (presynaptic partners). The float cast matters -- an int accumulator would
    overflow on the first hop out of a few hundred seed neurons.
    """
    adj = (W != 0).astype(np.float64).tocsr()
    if backward:
        adj = adj.T.tocsr()
    frontier = np.zeros(W.shape[0], dtype=bool)
    frontier[seeds] = True
    seen = frontier.copy()
    for _ in range(hops):
        nxt = ((adj @ frontier.astype(np.float64)) > 0) & ~seen
        seen |= nxt
        frontier = nxt
        if not nxt.any():
            break
    return seen


# ----------------------------------------------------------------- flywire arm

def load_flywire(n_obs, data_dir="data", min_syn=5, hops=2, full=False,
                 n_port_neurons=200, n_readout=64, seed=0):
    """The real connectome, behind the same `Connectome` interface.

    Schema is Codex's FlyWire export as of 2026-08 (see explore/schema_notes.md).
    Column names drift between releases -- a KeyError here is expected
    maintenance: print df.columns and fix the names.

      min_syn  edges below this synapse count are dropped. The Princeton file is
               NOT pre-thresholded (min is 1), so this is doing real work.
      hops     v1 subgraph: keep only neurons that sit on a port -> readout path
               within this many hops each way (plus the ports and readout
               themselves). `full=True` keeps all ~139k.
    """
    import pandas as pd

    conn = pd.read_csv(f"{data_dir}/connections_princeton.csv.gz")
    cls = pd.read_csv(f"{data_dir}/classification.csv.gz")

    # neuron universe = everything in the classification file, in root_id order
    cls = cls.sort_values("root_id").reset_index(drop=True)
    root_ids = cls["root_id"].to_numpy()
    pos = pd.Series(np.arange(len(root_ids)), index=root_ids)   # root_id -> index
    n = len(root_ids)

    # sign per PRESYNAPTIC neuron. nt_type is consistent per pre neuron in this
    # release (verified: 0 of 137518 disagree), so first() is exact, not a vote.
    nt = conn.drop_duplicates("pre_root_id").set_index("pre_root_id")["nt_type"]
    pre_sign = nt.map(NT_SIGN).fillna(0.0)

    # threshold, then collapse the per-neuropil rows into one edge per pair
    e = conn[conn["syn_count"] >= min_syn]
    e = (e.groupby(["pre_root_id", "post_root_id"], sort=False)["syn_count"]
           .sum().reset_index())
    e = e[e["pre_root_id"].isin(pos.index) & e["post_root_id"].isin(pos.index)]

    w = e["syn_count"].to_numpy(np.float64) * pre_sign.reindex(e["pre_root_id"]).to_numpy()
    keep = w != 0.0                        # drops DA / SER / OCT presynaptic cells
    i = pos.reindex(e["post_root_id"][keep]).to_numpy()
    j = pos.reindex(e["pre_root_id"][keep]).to_numpy()
    W = sp.coo_matrix((w[keep], (i, j)), shape=(n, n)).tocsr()
    W.sum_duplicates()

    # per-type params group by super_class/class (43 keys, NaN-safe), never
    # cell_type -- that column is not even in this release. See CLAUDE.md.
    key = cls["super_class"].fillna("NA") + "/" + cls["class"].fillna("NA")
    type_id = pd.factorize(key)[0].astype(np.int32)

    # readout: descending neurons. There are 1305; a linear map off all of them
    # is ~4k trained params and ES scales badly with dimension, so v1 samples a
    # fixed n_readout of them. Biologically arbitrary, fine for a first pass.
    desc = np.where((cls["super_class"] == "descending").to_numpy())[0]
    rng = np.random.default_rng(seed)
    out_idx = np.sort(rng.permutation(desc)[:n_readout])

    # ports: v1 -- arbitrary sensory neurons, split into one slab per channel.
    # Upgrade (step 3 note): map a retinotopic patch onto real medulla columns
    # using column_assignment.csv.gz.
    sens = np.where(cls["super_class"]
                    .isin(["sensory", "sensory_ascending"]).to_numpy())[0]
    sens = np.sort(rng.permutation(sens)[:max(n_obs, n_port_neurons)])
    ports = [np.asarray(a, np.int32) for a in np.array_split(sens, n_obs)]

    if not full:
        # keep the neurons that actually relay a port to the readout: forward
        # from the ports AND backward from the readout, both within the hop
        # budget. The intersection is small enough to eyeball; the raw forward
        # cone is most of the brain by hop 3.
        port_seeds = np.concatenate(ports)
        fwd = _reach(W, port_seeds, hops)
        bwd = _reach(W, out_idx, hops, backward=True)
        seen = fwd & bwd
        seen[port_seeds] = True
        seen[out_idx] = True
        sub = np.where(seen)[0]
        remap = np.full(n, -1, np.int64)
        remap[sub] = np.arange(sub.size)
        W = W[sub][:, sub]
        type_id = pd.factorize(type_id[sub])[0].astype(np.int32)  # recontiguous
        out_idx = remap[out_idx][remap[out_idx] >= 0]
        ports = [remap[p][remap[p] >= 0] for p in ports]

    W = normalise_incoming(W)
    return Connectome(W=W, type_id=type_id, ports=ports, out_idx=out_idx)


# ------------------------------------------------------- control arm 2: rewire

def rewire_degree_preserving(cx, n_swaps_per_edge=10, seed=0):
    """Step 5, arm 2: scramble the topology, keep everything else.

    Directed double-edge swap: pick edges a->b and c->d, replace with a->d and
    c->b. Every node keeps its exact in-degree and out-degree, and each weight
    stays attached to its original PRESYNAPTIC node -- so the per-neuron sign
    (Dale) and the pre-normalisation weight multiset are untouched. Only *who
    wires to whom* changes. `normalise_incoming` then runs exactly as it does
    for the real connectome, so post-normalisation magnitudes differ (different
    in-edges per row) -- that rescaling is applied identically to both arms.

    This is the control that answers "did the fly's wiring do something, or would
    any sparse graph with these degrees and weights do as well?". `make_synthetic`
    (arm 3) is the looser null that does not even match the degree sequence.

    Ports, readout, type_id, node identity: all unchanged.
    """
    rng = np.random.default_rng(seed)
    W = cx.W.tocoo()
    post = W.row.copy()          # b: presynaptic sign lives with `pre`, so we
    pre = W.col.copy()           #    only ever shuffle the `post` endpoints
    data = W.data.copy()
    m = len(data)

    existing = set(zip(pre.tolist(), post.tolist()))
    target = n_swaps_per_edge * m
    done = 0
    attempts = 0
    while done < target and attempts < target * 20:
        attempts += 1
        e1, e2 = rng.integers(0, m), rng.integers(0, m)
        if e1 == e2:
            continue
        a, b = pre[e1], post[e1]
        c, d = pre[e2], post[e2]
        if a == d or c == b:                       # would make a self-loop
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.discard((a, b))
        existing.discard((c, d))
        existing.add((a, d))
        existing.add((c, b))
        post[e1], post[e2] = d, b
        done += 1

    W2 = sp.coo_matrix((data, (post, pre)), shape=W.shape).tocsr()
    W2 = normalise_incoming(W2)
    return Connectome(W=W2, type_id=cx.type_id, ports=cx.ports,
                      out_idx=cx.out_idx)


# --------------------------------------------------------------------- smoke test

if __name__ == "__main__":
    import sys
    from snake import FEATURE_NAMES

    which = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    n_obs = len(FEATURE_NAMES)   # or 7*7*3 = 147 to rehearse the retina wiring

    if which == "flywire":
        cx = load_flywire(n_obs=n_obs, full="--full" in sys.argv)
    else:
        cx = make_synthetic(n_obs=n_obs, seed=0)
    print(cx.summary())

    port_seeds = np.concatenate([np.asarray(p) for p in cx.ports])
    for h in (1, 2, 3, 4, 6):
        s = _reach(cx.W, port_seeds, hops=h)
        print(f"  <= {h} hops from a port: {s.mean():5.1%} of net, "
              f"{s[cx.out_idx].mean():5.1%} of readout")
