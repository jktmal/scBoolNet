"""Shared TI contract export: Q1-Q3 observation selection, BoNesis constraints, slim h5ad.

Ported from `4. Trajectory Inference.ipynb` (originally notebook-local) so
`4. RNA Velocity.ipynb` can produce the same contract artifacts from a
CellRank-derived trajectory (`obs['pseudotime']`, `obs['branch']`,
`uns['root_state'/'terminal_states'/'transient_states']`) instead of PAGA+DPT.

See docs/chat-notes-veloBench.md §15 (recommendation) / §16 (CellRank pseudotime).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

from utils.trajectory_graph import derive_trajectory_graph

DEFAULT_UNS_KEYS = (
    "root_state",
    "terminal_states",
    "transient_states",
    "transient_order",
    "trajectory_edges",
    "root_cell",
    "iroot",
    "method",
    "ti_priors",
    "ti_priors_B",
    "paga",
    "cellrank_pseudotime_method",
)
DEFAULT_OBSM_KEYS = ("X_plot", "X_pca", "X_umap", "X_diffmap")


def select_observations(
    adata: ad.AnnData,
    *,
    pt_key: str = "pseudotime",
    label_key: str = "label",
) -> pd.Series:
    """Per phenotype: cells with pseudotime in [Q1, Q3] of that phenotype."""
    if pt_key not in adata.obs:
        raise KeyError(f"Missing obs[{pt_key!r}] — run trajectory inference first")
    if label_key not in adata.obs:
        raise KeyError(f"Missing obs[{label_key!r}]")

    parts = []
    for state, idx in adata.obs.groupby(label_key, observed=True).groups.items():
        state = str(state)
        pt = adata.obs.loc[idx, pt_key].astype(float)
        pt = pt[np.isfinite(pt)]
        if pt.empty:
            continue
        q1, q3 = pt.quantile(0.25), pt.quantile(0.75)
        mid = pt[(pt >= q1) & (pt <= q3)]
        if mid.empty:
            continue
        parts.append(pd.Series(state, index=mid.index, name="observation"))

    if not parts:
        raise ValueError("select_observations: no cells survived Q1-Q3 filter")
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = None
    return out


def derive_bonesis_constraints(
    adata: ad.AnnData,
    observations: pd.Series,
    *,
    method: str,
    pt_key: str = "pseudotime",
    label_key: str = "label",
    terminal_states: Optional[Sequence[str]] = None,
) -> dict:
    """BoNesis constraints from the states present in `observations`.

    Topology (which state is the immediate "hub" feeding a given terminal)
    comes from `utils.trajectory_graph.derive_trajectory_graph` — the real
    PAGA tree once that module is synced from its other-machine source, a
    median-pseudotime chain fallback until then (see its docstring) — kept
    consistent with `plot_state_graph` / `plot_streamlike`, which use the
    same function.
    """
    present = sorted(observations.astype(str).unique())
    terminals_prior = list(
        terminal_states if terminal_states is not None else adata.uns.get("terminal_states", [])
    )
    terminals = [s for s in terminals_prior if s in present]
    if not terminals:
        terminals = [s for s in present if s.startswith("p") and s[1:2].isupper()]

    root = str(adata.uns.get("root_state"))
    tree, root, edge_source = derive_trajectory_graph(
        adata, root=root, state_key=label_key, pt_key=pt_key, terminal_states=terminals
    )
    sub = tree.subgraph([n for n in tree.nodes if n in present or n == root])

    non_terminal = [s for s in present if s not in terminals]
    depth = nx.shortest_path_length(sub, root) if root in sub else {}
    med = (
        adata.obs.loc[observations.index]
        .groupby(observations.rename("observation"), observed=True)[pt_key]
        .median()
        .astype(float)
    )
    # order by real tree depth from root first, median pt only as a tie-break
    transient_order = sorted(
        non_terminal,
        key=lambda s: (depth.get(s, len(present) + 1), float(med.get(s, np.inf))),
    )
    hub = transient_order[-1] if transient_order else None

    def _real_hub(terminal: str) -> Optional[str]:
        """Terminal's actual tree predecessor when it's a known transient/root,
        else fall back to the global hub (keeps old behaviour for degenerate trees)."""
        preds = list(sub.predecessors(terminal)) if terminal in sub else []
        if preds and (preds[0] in transient_order or preds[0] == root):
            return preds[0]
        return hub

    bonesis_lines = []
    edges = []

    if len(transient_order) >= 2:
        chain = " >= ".join(f'~bo.obs("{s}")' for s in transient_order)
        bonesis_lines.append(chain)
        edges.extend(list(zip(transient_order[:-1], transient_order[1:])))

    for t in terminals:
        src = _real_hub(t)
        if src is None:
            continue
        bonesis_lines.append(f'~bo.obs("{src}") >= bo.fixed(~bo.obs("{t}"))')
        edges.append([src, t])

    supports_branching = len(terminals) >= 1 and (hub is not None or root in present)

    payload = {
        "method": method,
        "transient_chain": transient_order,
        "terminals": terminals,
        "hub": hub,
        "bonesis_lines": bonesis_lines,
        "edges": [list(e) for e in edges],
        "supports_branching": supports_branching,
        "states_in_csv": present,
        "edge_source": edge_source,
    }
    adata.uns["transient_order"] = transient_order
    adata.uns["trajectory_edges"] = payload["edges"]
    return payload


def write_bonesis_constraints(constraints: dict, results_dir: Path) -> tuple[Path, Path]:
    """Write `transition_constraints.{txt,json}` under `results_dir`. Returns (txt_path, json_path)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    txt_path = results_dir / "transition_constraints.txt"
    json_path = results_dir / "transition_constraints.json"
    txt_path.write_text(
        "\n".join(constraints["bonesis_lines"]) + ("\n" if constraints["bonesis_lines"] else "")
    )
    json_path.write_text(json.dumps(constraints, indent=2))
    return txt_path, json_path


def write_trajectory_h5ad(
    adata: ad.AnnData,
    out_path: Path,
    *,
    method: str,
    uns_keys: Sequence[str] = DEFAULT_UNS_KEYS,
    obsm_keys: Sequence[str] = DEFAULT_OBSM_KEYS,
) -> ad.AnnData:
    """Slim AnnData with contract slots only (obs + a few obsm/uns), written to `out_path`.

    Skips the full expression matrix — same as TI's `trajectory.h5ad`.
    """
    if "X_plot" not in adata.obsm:
        if "X_umap" in adata.obsm:
            adata.obsm["X_plot"] = np.asarray(adata.obsm["X_umap"])
        elif "X_pca" in adata.obsm:
            adata.obsm["X_plot"] = np.asarray(adata.obsm["X_pca"])[:, :2]

    adata.uns["method"] = method
    obsm_present = [k for k in obsm_keys if k in adata.obsm]
    ad_ti = ad.AnnData(
        obs=adata.obs.copy(),
        obsm={k: np.asarray(adata.obsm[k]) for k in obsm_present},
    )
    ad_ti.uns.update({k: adata.uns[k] for k in uns_keys if k in adata.uns})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ad_ti.write_h5ad(out_path)
    return ad_ti
