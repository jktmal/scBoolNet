"""PKN for BoNesis from a curated database (CollecTRI) and per-phenotype observations.

"Transcription factors" (TFs) are the regulators of the database (its edge
sources). Two variants, chosen with `core`:

- ``core=None`` — the rule of the authors' case study 1 code (Chevalier et al.
  2025, Zenodo 10.5281/zenodo.16990754): every TF -> TF edge of the database,
  also between TFs absent from the data, plus TF -> target edges for targets
  present in the data. TFs without observations act as free, hidden mediators.
- ``core="scc"`` — data genes plus the largest strongly connected component of
  the whole database, edges among them (scBoolSeq, Magaña-López et al. 2024).

``include_unmeasured_tfs=False`` keeps only data genes in either variant —
every node constrained by data (stricter than any published variant).

See src/grn_inference/README.md for the rationale and the numbers behind it.
"""

from typing import Optional

import networkx as nx
import pandas as pd

from grn_inference.grn import df_to_graph

# CollecTRI's `sign_decision` for edges whose sign has no literature support
# and was set to activation by default.
DEFAULT_SIGN_DECISION = "default activation"


def measured_genes(meta: pd.DataFrame, *, min_determined: int = 1) -> pd.Index:
    """Genes (columns of a phenotype x gene {0, 1, NaN} frame) with at least
    `min_determined` determined values across phenotypes (0 = every gene)."""
    return meta.columns[meta.notna().sum() >= min_determined]


def select_genes(
    meta: pd.DataFrame,
    net: pd.DataFrame,
    *,
    min_determined: int = 1,
    extra_genes=(),
    connect=(),
) -> pd.Index:
    """Genes of `meta` for a PKN restricted to them (`build_pkn(meta[genes], ...,
    include_unmeasured_tfs=False)`): genes with >= `min_determined` determined
    values plus `extra_genes` (e.g. HVG, markers).

    Genes of `connect` (e.g. markers) without a database edge to the rest of the
    selection get their direct database neighbours (regulators and targets)
    present in `meta`, otherwise they would drop out of the PKN. Genes absent
    from `meta` are left out.
    """
    present = set(meta.columns)
    genes = (set(measured_genes(meta, min_determined=min_determined)) | set(extra_genes)) & present
    for gene in set(connect) & genes:
        neighbours = set(net.target[net.source == gene]) | set(net.source[net.target == gene])
        neighbours.discard(gene)
        if not neighbours & genes:
            genes |= neighbours & present
    return meta.columns[meta.columns.isin(genes)]


def collectri_signed_edges(net: pd.DataFrame, *, default_sign_unknown: bool = True) -> pd.DataFrame:
    """One `source, target, sign` row per pair, sign in {1, -1, 0}.

    `sign` comes from CollecTRI's `weight`. With `default_sign_unknown`, edges
    whose `sign_decision` is "default activation" get sign 0, which BoNesis
    encodes as "either sign" (`in(a, b, (-1;1))`) and lets the solver pick per
    model. Duplicate pairs with conflicting signs also get 0.
    """
    edges = net[["source", "target"]].copy()
    edges["sign"] = net["weight"].apply(lambda w: (w > 0) - (w < 0)).astype(int)
    if default_sign_unknown and "sign_decision" in net:
        edges.loc[net["sign_decision"] == DEFAULT_SIGN_DECISION, "sign"] = 0
    return (
        edges.groupby(["source", "target"], as_index=False)["sign"]
        .agg(lambda s: s.iloc[0] if s.nunique() == 1 else 0)
    )


def build_pkn(
    net: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    core: Optional[str] = None,
    min_determined: int = 0,
    include_unmeasured_tfs: bool = True,
    largest_wcc: bool = False,
    default_sign_unknown: bool = True,
) -> tuple[nx.MultiDiGraph, dict]:
    """PKN from database edges (`net`: source/target/weight[/sign_decision]) and
    the data genes of `meta` (genes with >= `min_determined` determined values;
    0 = every gene of `meta`, i.e. presence in the data, as in case study 1).

    - ``core=None``: TF -> TF edges + TF -> data-gene edges (case study 1).
    - ``core="scc"``: edges among data genes and the database's largest SCC.

    `include_unmeasured_tfs=False` drops every node that isn't a data gene.
    Nodes without edges are left out (self-loops count as edges). With
    `largest_wcc`, only the largest weakly connected component is kept.
    The scBoolSeq-style PKN of the previous `5. GRN Construction` is
    ``core="scc", largest_wcc=True, default_sign_unknown=False``.

    Nodes without inputs are returned as they are; case study 1 loaded its PKN
    with `bonesis.InfluenceGraph.from_sif`, which adds a positive self-loop to
    each of them — call `.unsource()` on the InfluenceGraph to do the same.

    Returns (graph, report). `graph` is a networkx MultiDiGraph with an int `sign`
    attribute per edge — wrap it in `bonesis.InfluenceGraph` for solving.
    """
    if core not in (None, "scc"):
        raise ValueError(f"core must be None or 'scc', got {core!r}")

    data_genes = measured_genes(meta, min_determined=min_determined)
    data = set(data_genes)
    observed = set(measured_genes(meta))  # >= 1 determined value, for the report
    edges = collectri_signed_edges(net, default_sign_unknown=default_sign_unknown)
    tfs = set(edges.source)

    core_nodes: set = set()
    if core is None:
        keep = edges.target.isin(tfs) | edges.target.isin(data)
    else:
        whole = nx.DiGraph(list(zip(edges.source, edges.target)))
        core_nodes = max(nx.strongly_connected_components(whole), key=len)
        nodes = data | core_nodes
        keep = edges.source.isin(nodes) & edges.target.isin(nodes)
    if not include_unmeasured_tfs:
        keep &= edges.source.isin(data) & edges.target.isin(data)
    edges = edges[keep]

    graph = df_to_graph(edges, source="source", target="target")
    if largest_wcc and graph.number_of_nodes():
        graph = graph.subgraph(max(nx.weakly_connected_components(graph), key=len)).copy()

    kept = pd.DataFrame(
        [(u, v, d["sign"]) for u, v, d in graph.edges(data=True)], columns=["source", "target", "sign"]
    )
    signs = kept["sign"].value_counts()
    simple = nx.DiGraph(graph)
    with_obs = sum(1 for n in graph if n in observed)
    tf_nodes = [n for n in graph if n in tfs]
    report = {
        "core": core,
        "core_size": len(core_nodes),
        "core_with_observations": len(core_nodes & observed),
        "genes_in_meta": int(meta.shape[1]),
        "data_genes": int(len(data_genes)),
        "data_genes_not_in_db": int((~data_genes.isin(pd.concat([net.source, net.target]))).sum()),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "regulators": sum(1 for n in graph if graph.out_degree(n) > 0),
        "tf_nodes": len(tf_nodes),
        "tfs_not_in_data": sum(1 for n in tf_nodes if n not in data),
        "nodes_with_observations": with_obs,
        "observed_fraction": float(with_obs / max(graph.number_of_nodes(), 1)),
        "nodes_without_inputs": sum(1 for _, d in graph.in_degree() if d == 0),
        "sign_counts": {int(k): int(v) for k, v in signs.items()},
        "unknown_sign_fraction": float(signs.get(0, 0) / max(len(kept), 1)),
        "largest_scc": max((len(c) for c in nx.strongly_connected_components(simple)), default=0),
        "largest_wcc": max((len(c) for c in nx.weakly_connected_components(simple)), default=0),
    }
    return graph, report
