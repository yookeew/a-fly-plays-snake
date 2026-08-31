# data/

Connectome files live here. They are gitignored — download your own copy.

## FlyWire (Codex)

Free account at https://codex.flywire.ai, then **Info → Download Data**.

| file                  | what it is                                                        |
| --------------------- | ----------------------------------------------------------------- |
| `connections.csv`     | `pre_root_id, post_root_id, neuropil, syn_count, nt_type`         |
| `classification.csv`  | `root_id, super_class, class, cell_type, side, …`                 |

Data is CC-BY 4.0. If you publish anything, follow the citation guidelines on
Codex — the connectivity, the annotations, and the neurotransmitter predictions
are separate papers and want separate citations.

Column names drift between releases. If a loader raises `KeyError`, print
`df.columns` and adjust. That's expected maintenance, not a bug.