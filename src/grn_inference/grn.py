""" Functions to manipulate Gene Regulatory Networks (GRNs)."""
import copy
import typing
import itertools

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl

__all__ = ["domain_union", "df_to_graph", "extract_largest_scc"]


# Graph Types
_Domain = nx.MultiDiGraph
_DomainIterable = typing.Iterable[_Domain]

class InfluenceGraphUnion(typing.NamedTuple):
    ig: _Domain
    contradictions: typing.List[typing.Tuple[str, str, int]]

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"ig = <{self.ig.__class__.__name__} with {len(self.ig.nodes)} nodes, {len(self.ig.edges)} edges> ; ",
                f"contradictions = <List of {len(self.contradictions)} edges>)",
            ]
        )

def domain_union(
    domains: _DomainIterable, display_mode: str = "proba", relabel_edges: bool = True
) -> InfluenceGraphUnion:
    """Take the union of two or more domains.
    The resulting domain will be a tuple:
        (MultiDiGraph, List[(str,str,int)])
    containing the union as well as a list of
    nodes that showed contradictive influences
    between the provided domains."""
    _valid_display_modes = ["proba", "vote"]
    if display_mode not in _valid_display_modes:
        raise ValueError(
            f"Invalid display mode `{display_mode}`.",
            f"Choose one of {_valid_display_modes}",
        )

    _aggregated_graph = nx.from_edgelist(
        itertools.chain.from_iterable((i.edges(data=True) for i in domains)),
        create_using=nx.MultiDiGraph,
    )

    multiedge_cardinals = {}
    for i, j, _n in _aggregated_graph.edges:
        _aggregated_graph.edges[(i, j, _n)]["vote"] = 1
        k = (i, j)
        if k not in multiedge_cardinals:
            multiedge_cardinals.update({k: [_n]})
        else:
            multiedge_cardinals[k].append(_n)

    redundant = {k: v for k, v in multiedge_cardinals.items() if len(v) > 1}
    contradictions = []

    for k in redundant:
        # ml stands for multi-label
        first = _aggregated_graph.edges[(*k, redundant[k][0])]
        ml_map = {first["label"]: first}
        _k_count = 0
        for j in redundant[k][1:]:
            step = (*k, j)
            step_edge = _aggregated_graph.edges[step]
            step_label = step_edge["label"]
            if step_label in ml_map:
                first["vote"] += 1
                _aggregated_graph.remove_edges_from([step])
            else:
                ml_map.update({step_label: step_edge})
                contradictions.append(step)
            _k_count += 1

    _total = len(domains)
    for start_node, end_node, contradiction in contradictions:
        v1 = _aggregated_graph.edges[(start_node, end_node, 0)]
        v2 = _aggregated_graph.edges[(start_node, end_node, contradiction)]
        # _total = v1["vote"] + v2["vote"]
        v1["proba"] = v1["vote"] / _total
        v2["proba"] = v2["vote"] / _total

    for i in _aggregated_graph.edges:
        _d = _aggregated_graph.edges[i]
        if "proba" not in _d:
            _d["proba"] = _d["vote"] / _total

        # Probabilities of 1.0 do not need to be shown
        if _d["proba"] < 1.0 and relabel_edges:
            if display_mode == "proba":
                _d["label"] = f"{_d['proba']:.2f} {_d['label']}"
            elif display_mode == "vote":
                _d["label"] = f"{_d['vote']} {_d['label']}"

    return InfluenceGraphUnion(_aggregated_graph, contradictions)


def df_to_graph(
    frame: pd.DataFrame, source: str = "tf", target: str = "target"
) -> nx.MultiDiGraph:
    """Transform a dataframe to a networkx.MultiDiGraph
    Default values are set to directly construct a GRN network from the
    DoRothEA: collection of human and mouse regulons.

    For more information on DoRothEA, see:
        https://saezlab.github.io/dorothea/index.html

    Args:
        df (pd.DataFrame): Describing a graph (gene regulatory network in our case).
        source (str, optional): Column name of directed edges' sources. Defaults to "tf".
        target (str, optional): Column name of directed edges' targets. Defaults to "target".

    Returns:
        nx.MultiDiGraph: A directed graph representing the information contained
                         in param `df`. Ideally a Gene Regulatory Network.
    """
    return nx.from_pandas_edgelist(
        frame,
        source=source,
        target=target,
        edge_attr=True,
        create_using=nx.MultiDiGraph,
    )


def extract_largest_scc(grn: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Extract a gene regulatory network's largest strongly connected component

    Args:
        grn (nx.MultiDiGraph): A GRN.

    Returns:
        nx.MultiDiGraph: The GRN's largest strongly connected component.
    """
    return nx.subgraph(grn, max(nx.strongly_connected_components(grn), key=len))

def extract_largest_wcc(grn: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Extract a gene regulatory network's largest weakly connected component

    Args:
        grn (nx.MultiDiGraph): A GRN.

    Returns:
        nx.MultiDiGraph: The GRN's largest weakly connected component.
    """
    return nx.subgraph(grn, max(nx.weakly_connected_components(grn), key=len))

def colorFader(c1, c2, mix=0):
    """fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    taken from:
    https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2) 

def color_influence_graph(ig, mode="influence", cold="lightblue", hot="yellow", scores=None):
    """
    available modes are:
    [
      in_degree, out_degree, influence,
      total_degree, norm_total_degree, square_total_degree
    ]

    `mode` can be a lambda function, using the influence graph attibutes:
    The parameter will be the node name.
    For example, "influence" is defined as
        lambda node: (ig.out_degree[node]) / (max(ig.in_degree[node],1))

    another option could be:
        mode=lambda xx: (ig.out_degree[xx]+1)**2 - ig.in_degree[xx]
    """

    ig = copy.deepcopy(ig)

    scoring_modes = {
        "in_degree": lambda node: ig.in_degree[node],
        "out_degree": lambda node: ig.out_degree[node],
        "influence": lambda node: (ig.out_degree[node]) / (max(ig.in_degree[node], 1)),
        "total_degree": lambda node: ig.in_degree[node] + ig.out_degree[node],
        "norm_total_degree": lambda node: (ig.out_degree[node] + ig.in_degree[node])
        / max(ig.in_degree[node], 1),
        "square_total_degree": lambda node: (
            ig.out_degree[node] ** 2 + ig.in_degree[node]
        )
        / max(ig.in_degree[node], 1),
    }
    _f_score = scoring_modes.get(mode)
    _f_score = _f_score or mode  # if mode is a callable

    if scores is not None:
        node_scores = scores
    else:
        node_scores = {_node: _f_score(_node) for _node in ig.nodes}
    highest_rank = max(node_scores.values())
    node_colour_map = {
        _node: colorFader(cold, hot, _score / highest_rank)
        for _node, _score in sorted(node_scores.items(), key=lambda _tuple: _tuple[1])
    }

    for node, data in ig.nodes(data=True):
        data["fillcolor"] = node_colour_map[node]
        data["style"] = "filled"

    return ig
