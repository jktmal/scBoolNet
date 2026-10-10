"""Trajectory state-graph topology (§12.5 / §12.12 contract).

Derives a directed state tree oriented from ``root_state``, preferring the
*real* group graph a method already computed (currently: PAGA's
``connectivities_tree``) over inventing one from a 1-D pseudotime ranking.
A pseudotime-median chain is kept only as a fallback for methods that don't
produce a group graph at all.

Used by the "flat-tree-like" state graph, ``derive_bonesis_constraints``, and
(via ``adata.uns['trajectory_edges']``) ``plot_streamlike``, so all three
agree on the same topology.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def _paga_group_graph(adata: AnnData, state_key: str) -> nx.Graph | None:
    """Undirected group graph from ``uns['paga']['connectivities_tree']``.

    ``connectivities_tree`` is scanpy's own maximum-spanning-tree over the
    PAGA connectivities (``scipy.sparse.csgraph.minimum_spanning_tree`` on
    inverse connectivities), so it already spans every group in ``groups`` —
    the same graph ``sc.pl.paga`` / ``paga_compare`` draw. Returns ``None``
    when PAGA hasn't been run (or wasn't run on ``state_key``).
    """
    paga = adata.uns.get("paga")
    if not paga or "connectivities_tree" not in paga:
        return None
    if state_key not in adata.obs or not hasattr(adata.obs[state_key], "cat"):
        return None
    categories = [str(c) for c in adata.obs[state_key].cat.categories]
    tree = paga["connectivities_tree"]
    tree = tree.toarray() if sp.issparse(tree) else np.asarray(tree)
    if tree.shape != (len(categories), len(categories)):
        return None
    tree = np.maximum(tree, tree.T)  # MST rows are stored one-directional
    G = nx.Graph()
    G.add_nodes_from(categories)
    n = len(categories)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(tree[i, j])
            if w > 0:
                G.add_edge(categories[i], categories[j], weight=w)
    return G


def _pseudotime_chain_graph(
    adata: AnnData,
    *,
    state_key: str,
    pt_key: str,
    root: str,
    terminal_states: Sequence[str] | None,
) -> nx.Graph:
    """Fallback when no group graph is available: chain non-terminal states
    by median pseudotime, then attach each terminal to the last (highest-pt)
    non-terminal. Cruder than a real graph (can misorder states whose
    pseudotime isn't monotonic with real topology) — kept only so a method
    without a group graph still produces *some* tree.
    """
    med = (
        adata.obs.groupby(state_key, observed=True)[pt_key]
        .median()
        .astype(float)
        .sort_values()
    )
    states = [str(s) for s in med.index]
    # `terminal_states` may be a numpy array here (anndata round-trips list
    # uns values through h5ad as ndarrays) — `x or []` on a multi-element
    # array raises ValueError, so guard on `is not None` instead.
    terminals = set(map(str, terminal_states if terminal_states is not None else []))
    non_term = [s for s in states if s not in terminals]
    if root in non_term:
        non_term = [root] + [s for s in non_term if s != root]
    term = [s for s in states if s in terminals]

    G = nx.Graph()
    G.add_nodes_from(states)
    for u, v in zip(non_term[:-1], non_term[1:]):
        G.add_edge(u, v, weight=1.0)
    hub = non_term[-1] if non_term else root
    for t in term:
        if t != hub:
            G.add_edge(hub, t, weight=1.0)
    return G


def derive_trajectory_graph(
    adata: AnnData,
    *,
    root: str | None = None,
    state_key: str = "label",
    pt_key: str = "pseudotime",
    terminal_states: Sequence[str] | None = None,
) -> tuple[nx.DiGraph, str, str]:
    """Directed state tree oriented away from ``root``.

    Returns
    -------
    tree : nx.DiGraph
        Edges oriented from ``root`` outward (a BFS tree).
    root : str
    source : str
        ``"paga"`` or ``"pseudotime_chain"`` — which graph was actually used.
    """
    root = str(root or adata.uns.get("root_state") or "")
    if not root:
        raise ValueError("root / uns['root_state'] is required")
    if state_key not in adata.obs:
        raise KeyError(f"Missing obs[{state_key!r}]")

    present = set(adata.obs[state_key].dropna().astype(str).unique())

    G = _paga_group_graph(adata, state_key)
    source = "paga"
    if G is not None:
        G = G.subgraph([n for n in G.nodes if n in present]).copy()

    if G is None or root not in G or set(G.nodes) != present or not nx.is_connected(G):
        G = _pseudotime_chain_graph(
            adata,
            state_key=state_key,
            pt_key=pt_key,
            root=root,
            terminal_states=terminal_states,
        )
        source = "pseudotime_chain"

    if root not in G:
        raise ValueError(f"root {root!r} not present in state graph")

    mst = G if nx.is_tree(G) else nx.maximum_spanning_tree(G, weight="weight")
    tree = nx.bfs_tree(mst, source=root)
    return tree, root, source
