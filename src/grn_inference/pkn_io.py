"""Export/import contract for the PKN + observations handed off between
"5. GRN Construction.ipynb" (or an alternative PKN source, e.g. FlowGRN) and
"6. BN Inference (BoNesis).ipynb", round-tripped through JSON so the two
notebooks don't need to share a Python process.
"""

import json

import bonesis
import networkx as nx
import pandas as pd


def df_to_bonesis_data(df: pd.DataFrame) -> dict:
    """Convert a phenotype x gene binarized dataframe into bonesis' observation dict format."""
    data = {}
    for config, genes in df.iterrows():
        data[config] = genes.dropna().to_dict()
    return data


def export_pkn_bundle(file_prefix: str, prior_grn, pkn, data_meta: dict, include_hc_constraint: bool) -> None:
    """Export the full PKN, the solver-facing PKN, observations and run config to disk."""
    with open(f"{file_prefix}_prior_grn.json", "w") as f:
        json.dump(nx.node_link_data(prior_grn), f)

    with open(f"{file_prefix}_pkn.json", "w") as f:
        json.dump(nx.node_link_data(pkn), f)

    with open(f"{file_prefix}_data_meta.json", "w") as f:
        json.dump(data_meta, f)

    with open(f"{file_prefix}_config.json", "w") as f:
        json.dump({"INCLUDE_HC_CONSTRAINT": include_hc_constraint}, f)


def load_pkn_bundle(
    file_prefix: str,
    maxclause: int = 8,
    allow_skipping_nodes: bool = True,
    canonic: bool = False,
):
    """Load a bundle written by `export_pkn_bundle`, re-wrapping `pkn` as a bonesis InfluenceGraph.

    `maxclause`/`allow_skipping_nodes`/`canonic` are plain Python attributes on
    `bonesis.domains.InfluenceGraph`, not part of the networkx graph data, so they
    don't survive `nx.node_link_data` — they must be passed again here, matching
    whatever the PKN was originally built with.

    Returns (prior_grn, pkn, data_meta, include_hc_constraint).
    """
    with open(f"{file_prefix}_prior_grn.json") as f:
        prior_grn = nx.node_link_graph(json.load(f))

    with open(f"{file_prefix}_pkn.json") as f:
        pkn_graph = nx.node_link_graph(json.load(f))

    with open(f"{file_prefix}_data_meta.json") as f:
        data_meta = json.load(f)

    with open(f"{file_prefix}_config.json") as f:
        config = json.load(f)

    pkn = bonesis.domains.InfluenceGraph(
        pkn_graph,
        maxclause=maxclause,
        allow_skipping_nodes=allow_skipping_nodes,
        canonic=canonic,
    )
    return prior_grn, pkn, data_meta, config["INCLUDE_HC_CONSTRAINT"]
