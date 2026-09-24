"""Shared trajectory-inference plots — used by both `4. Trajectory Inference.ipynb`
(PAGA+DPT) and `4. RNA Velocity.ipynb` (CellRank) so the two pipelines produce
visually comparable diagnostics from the same contract fields
(`obs['pseudotime']`, `obs['branch']`, `uns['root_state'/'terminal_states']`).

Only plots that are more than a few lines live here — a single `sc.pl.embedding`
or `sc.pl.paga` call stays inline in the notebooks.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

from utils.trajectory_graph import derive_trajectory_graph


def plot_pseudotime_boxplot(
    adata: ad.AnnData,
    *,
    pseudotime_key: str = "pseudotime",
    branch_key: str = "branch",
    ax: Optional[plt.Axes] = None,
    ylabel: Optional[str] = None,
) -> tuple[plt.Axes, pd.Series]:
    """Boxplot of pseudotime per branch, ordered by median (root -> terminal)."""
    order = (
        adata.obs.groupby(branch_key, observed=True)[pseudotime_key]
        .median()
        .sort_values()
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    data = [
        adata.obs.loc[adata.obs[branch_key] == b, pseudotime_key].to_numpy()
        for b in order.index
    ]
    ax.boxplot(data, tick_labels=list(order.index), vert=True, showfliers=False)
    ax.set_xticklabels(list(order.index), rotation=45, ha="right")
    ax.set_ylabel(ylabel or pseudotime_key)
    return ax, order


def plot_pseudotime_kde(
    adata: ad.AnnData,
    *,
    pseudotime_key: str = "pseudotime",
    root_cell: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """KDE of pseudotime with Q1/Q3 markers (+ root cell line, if given)."""
    pt = adata.obs[pseudotime_key].astype(float)
    pt = pt[np.isfinite(pt)]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))
    pt.plot.kde(ax=ax, label="all cells")
    q1, q3 = float(pt.quantile(0.25)), float(pt.quantile(0.75))
    ax.axvline(q1, color="red", ls="--", label=f"Q1={q1:.3g}")
    ax.axvline(q3, color="red", ls="--", label=f"Q3={q3:.3g}")
    if root_cell is not None and root_cell in adata.obs_names:
        ax.axvline(
            float(adata.obs.loc[root_cell, pseudotime_key]),
            color="black",
            ls=":",
            label="root_cell",
        )
    ax.set_xlabel(pseudotime_key)
    ax.set_title(f"KDE {pseudotime_key}")
    ax.legend(fontsize=8)
    return ax


def plot_state_graph(
    adata: ad.AnnData,
    *,
    pseudotime_key: str = "pseudotime",
    branch_key: str = "branch",
    root_state: Optional[str] = None,
    terminal_states: Optional[Sequence[str]] = None,
    palette: Optional[Mapping[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Axes, list]:
    """Flat-tree-like state graph: nodes = branch, x = median pseudotime.

    Edges come from `utils.trajectory_graph.derive_trajectory_graph` (real PAGA
    tree once synced, median-pseudotime chain fallback until then — see that
    module's docstring), same function `derive_bonesis_constraints` uses.

    `palette` (e.g. `adata.uns['label_color']`, a state -> hex color dict):
    when given, nodes are filled with the state's own canonical color and
    root/terminal are marked by a black/amber border instead (matching TI's
    flat-tree cell); when omitted (default), falls back to the plain
    red/green/blue root/terminal/other scheme used so far — no behavior change
    for existing callers.
    """
    med = (
        adata.obs.groupby(branch_key, observed=True)[pseudotime_key]
        .median()
        .astype(float)
        .sort_values()
    )
    counts = adata.obs[branch_key].astype(str).value_counts()

    terminal_set = set(map(str, terminal_states if terminal_states is not None else []))
    root = str(root_state) if root_state is not None else None
    states = [str(s) for s in med.index]
    non_term = [s for s in states if s not in terminal_set]
    term = [s for s in states if s in terminal_set]

    G, root, _edge_source = derive_trajectory_graph(
        adata, root=root, state_key=branch_key, pt_key=pseudotime_key, terminal_states=terminal_states
    )
    edges = list(G.edges())

    pos = {s: (float(med[s]), 0.0) for s in non_term}
    if term:
        ys = np.linspace(-1, 1, num=len(term)) if len(term) > 1 else [0.0]
        for t, y in zip(term, ys):
            pos[t] = (float(med[t]), float(y))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.5)
    sizes = [200 + 8 * int(counts.get(s, 0) ** 0.5) for s in G.nodes]

    if palette:
        node_fill = [palette.get(s, "#1f77b4") for s in G.nodes]
        node_edge = [
            "black" if s == root else ("#b8860b" if s in terminal_set else "white")
            for s in G.nodes
        ]
        node_lw = [2.5 if (s == root or s in terminal_set) else 0.0 for s in G.nodes]
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_fill, edgecolors=node_edge, linewidths=node_lw,
            node_size=sizes, alpha=0.95,
        )
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                       markeredgecolor="black", markeredgewidth=2.5, markersize=9, label="root"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                       markeredgecolor="#b8860b", markeredgewidth=2.5, markersize=9, label="terminal"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    else:
        node_colors = [
            "#d62728" if s == root else ("#2ca02c" if s in terminal_set else "#1f77b4")
            for s in G.nodes
        ]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=sizes, alpha=0.9)

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    ax.set_xlabel(f"median {pseudotime_key}")
    ax.set_yticks([])
    ax.set_title("State graph (flat-tree-like)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return ax, edges


def plot_marker_genes(
    adata: ad.AnnData,
    markers: Mapping[str, Sequence[str]],
    *,
    priority_states: Optional[Sequence[Optional[str]]] = None,
    max_genes: int = 8,
    basis: str = "umap",
    ncols: int = 4,
    **kwargs,
) -> list[str]:
    """Embedding colored by marker genes, prioritized by `priority_states` order.

    `markers` maps a state name to its marker genes (e.g. `states_and_markers`
    from `markers.py` in the TI notebook) — passed in rather than imported here,
    so this isn't tied to one dataset's marker registry. `priority_states`
    defaults to `markers`' own key order; pass e.g.
    `[root_state, *terminal_states, *transient_states]` to prioritize like TI does.

    Returns the gene list actually plotted (empty if none of the candidate
    genes are in `adata.var_names` — caller decides what to do about that,
    this function doesn't print or skip on your behalf).
    """
    priority: list[str] = []
    for state in priority_states if priority_states is not None else list(markers):
        if state and state in markers:
            priority.extend(sorted(markers[state]))

    genes = [g for g in dict.fromkeys(priority) if g in adata.var_names][:max_genes]
    if not genes:
        return []

    sc.pl.embedding(adata, basis=basis, color=genes, ncols=ncols, **kwargs)
    return genes


def plot_pseudotime_comparison(
    adata: ad.AnnData,
    x_key: str,
    y_key: str,
    *,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Axes, float]:
    """Scatter of two pseudotime-like `obs` columns + their Spearman correlation.

    Generic QC for comparing any two pseudotime-like estimates on the same
    cells — e.g. a paper's ground truth vs DPT (as in TI), or CellRank's
    absorption-time pseudotime vs scVelo's `velocity_pseudotime` (as in RNA
    Velocity) — whichever `x_key`/`y_key` `obs` columns you give it. Raises
    `KeyError` if either column is missing; check for optional columns (like
    an external ground truth that may not exist) before calling.
    """
    if x_key not in adata.obs:
        raise KeyError(f"Missing obs[{x_key!r}]")
    if y_key not in adata.obs:
        raise KeyError(f"Missing obs[{y_key!r}]")

    x = adata.obs[x_key].astype(float)
    y = adata.obs[y_key].astype(float)
    mask = np.isfinite(x) & np.isfinite(y)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(x[mask], y[mask], s=3, alpha=0.25, c="0.3")
    ax.set_xlabel(x_label or x_key)
    ax.set_ylabel(y_label or y_key)
    rho = float(pd.DataFrame({x_key: x[mask], y_key: y[mask]}).corr(method="spearman").iloc[0, 1])
    ax.set_title(f"Spearman ρ = {rho:.3f}")
    return ax, rho
