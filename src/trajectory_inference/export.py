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


def _contract_to_present(tree: nx.DiGraph, root: str, present: set[str]) -> nx.DiGraph:
    """Drop states absent from the observations, linking each kept state to its
    nearest kept ancestor (so an empty intermediate state doesn't split the tree)."""
    if root not in present:
        raise ValueError(f"root {root!r} has no observations — cannot anchor BoNesis constraints")
    out = nx.DiGraph()
    out.add_node(root)
    for v in nx.topological_sort(tree):
        if v == root or v not in present:
            continue
        anc = next(iter(tree.predecessors(v)), None)
        while anc is not None and anc not in present:
            anc = next(iter(tree.predecessors(anc)), None)
        if anc is not None:
            out.add_edge(anc, v)
    return out


def bonesis_spec_from_tree(
    tree: nx.DiGraph, root: str, *, side_leaves: Sequence[str] = ()
) -> dict:
    """BoNesis specification from a rooted state tree (Chevalier et al. 2025, Fig. 2C).

    - terminals = leaves of `tree` minus `side_leaves`; each is a fixed point;
    - reachability along every tree edge, written as chains (one per branch);
    - commitment: from a non-terminal state, terminals outside its subtree are
      unreachable (`/`);
    - universal: from the root and from the first state of each new branch, every
      reachable fixed point matches one of the terminals of its subtree (`>> "fixpoints"`);
    - distinct: the fixed points of any two terminals are different configurations
      (`!=`), as in the authors' case study 1 code.

    `side_leaves` are leaves that are not cell fates (e.g. ``ifnHSC``): they only
    get reachability, no fixed point and no commitment/universal constraints.
    """
    side = {s for s in side_leaves if s in tree and tree.out_degree(s) == 0}
    terminals = sorted(v for v in tree if tree.out_degree(v) == 0 and v != root and v not in side)
    term_below = {v: ({v} | nx.descendants(tree, v)) & set(terminals) for v in tree}

    def _size(v: str) -> int:
        return len(nx.descendants(tree, v))

    # Chain decomposition: the largest child continues the chain, the others start new ones.
    chains, todo = [], [[root]]
    while todo:
        chain = todo.pop(0)
        while True:
            kids = sorted(tree.successors(chain[-1]), key=lambda k: (-_size(k), k))
            if not kids:
                break
            todo.extend([chain[-1], k] for k in kids[1:])
            chain.append(kids[0])
        if len(chain) > 1:
            chains.append(chain)

    nonreach, universal = [], []
    for v in nx.topological_sort(tree):
        if v in terminals or not term_below[v]:
            continue
        parent = next(iter(tree.predecessors(v)), None)
        if parent is None or term_below[v] != term_below[parent]:
            universal.append([v, sorted(term_below[v])])
        nonreach.extend([v, t] for t in terminals if t not in term_below[v])

    return {
        "root": root,
        "terminals": terminals,
        "side_leaves": sorted(side),
        "edges": [list(e) for e in tree.edges],
        "chains": chains,
        "nonreach": nonreach,
        "universal": universal,
        "distinct": [[a, b] for i, a in enumerate(terminals) for b in terminals[i + 1:]],
    }


def render_bonesis_lines(spec: dict) -> list[str]:
    """Human-readable BoNesis code for `spec` (same statements `apply_bonesis_constraints` issues)."""
    terminals = set(spec["terminals"])

    def ref(s: str) -> str:
        return f'fixed["{s}"]' if s in terminals else f'~bo.obs("{s}")'

    lines = ["fixed = {}"] if terminals else []
    lines += [f'fixed["{t}"] = bo.fixed(~bo.obs("{t}"))' for t in spec["terminals"]]
    lines += [f'fixed["{a}"] != fixed["{b}"]' for a, b in spec["distinct"]]
    lines += [" >= ".join(ref(s) for s in chain) for chain in spec["chains"]]
    lines += [f'~bo.obs("{v}") / fixed["{t}"]' for v, t in spec["nonreach"]]
    lines += [
        f'~bo.obs("{v}") >> "fixpoints" ^ {{' + ", ".join(f'bo.obs("{t}")' for t in ts) + "}"
        for v, ts in spec["universal"]
    ]
    return lines


def apply_bonesis_constraints(bo, spec: dict, *, negative: bool = True) -> None:
    """Declare `spec` on a `bonesis.BoNesis` object.

    `~bo.obs(s)` always denotes the same configuration (BoNesis names it after the
    observation), so all statements about a state constrain one shared configuration.
    `negative=False` skips nonreach/universal constraints — the paper's practice for
    the (costly) node-selection stage, re-adding them only for model sampling.
    Distinct fixed points (`!=`) are always declared.
    """
    states = {s for chain in spec["chains"] for s in chain} | set(spec["terminals"])
    missing = sorted(states - set(getattr(bo, "data", states)))
    if missing:
        raise KeyError(f"states without observations in bo.data: {missing}")

    fixed = {t: bo.fixed(~bo.obs(t)) for t in spec["terminals"]}

    def ref(s: str):
        return fixed[s] if s in fixed else ~bo.obs(s)

    for a, b in spec["distinct"]:
        fixed[a] != fixed[b]
    for chain in spec["chains"]:
        for u, v in zip(chain[:-1], chain[1:]):
            ~bo.obs(u) >= ref(v)
    if not negative:
        return
    for v, t in spec["nonreach"]:
        ~bo.obs(v) / fixed[t]
    for v, ts in spec["universal"]:
        ~bo.obs(v) >> "fixpoints" ^ {bo.obs(t) for t in ts}


def derive_bonesis_constraints(
    adata: ad.AnnData,
    observations: pd.Series,
    *,
    method: str,
    pt_key: str = "pseudotime",
    label_key: str = "label",
    terminal_states: Optional[Sequence[str]] = None,
    transient_states: Optional[Sequence[str]] = None,
) -> dict:
    """BoNesis constraints from the state tree restricted to states in `observations`.

    Topology comes from `utils.trajectory_graph.derive_trajectory_graph` (PAGA tree,
    median-pseudotime chain as fallback) — the same graph `plot_state_graph` /
    `plot_streamlike` draw. Terminals are the tree's leaves, except leaves listed in
    `transient_states` (default `uns['transient_states']`), which only get
    reachability — see `bonesis_spec_from_tree`. `terminal_states` (default
    `uns['terminal_states']`) only feeds the fallback graph and the
    `terminals_prior` diagnostic.
    """
    present = set(observations.astype(str).unique())
    terminals_prior = [
        str(s) for s in (terminal_states if terminal_states is not None else adata.uns.get("terminal_states", []))
    ]
    transient_prior = [
        str(s) for s in (transient_states if transient_states is not None else adata.uns.get("transient_states", []))
    ]

    tree, root, edge_source = derive_trajectory_graph(
        adata, root=adata.uns.get("root_state"), state_key=label_key, pt_key=pt_key,
        terminal_states=terminals_prior,
    )
    tree = _contract_to_present(tree, root, present)
    spec = bonesis_spec_from_tree(tree, root, side_leaves=transient_prior)

    payload = {
        "method": method,
        **spec,
        "bonesis_lines": render_bonesis_lines(spec),
        "terminals_prior": terminals_prior,
        "states_in_csv": sorted(present),
        "edge_source": edge_source,
    }
    adata.uns["transient_order"] = [s for s in nx.topological_sort(tree) if s not in spec["terminals"]]
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
