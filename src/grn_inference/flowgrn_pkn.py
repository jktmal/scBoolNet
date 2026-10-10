"""Build a data-driven PKN via FlowGRN, as an alternative to the curated
CollecTRI-based PKN in "5. GRN Construction.ipynb".

Pipeline (see "5. GRN Construction (FlowGRN).ipynb"):
1. `derive_branches` — walk `adata.uns["trajectory_edges"]` from `root_state` to
   each `terminal_state` (one FlowGRN "branch" per terminal), using the same
   trajectory contract produced by "4. Trajectory Inference.ipynb"
   (`adata.obs["pseudotime"]`/`["label"]`, `adata.uns["root_state"]`/
   `["terminal_states"]`/`["trajectory_edges"]`).
2. `write_beeline_inputs` — one combined BEELINE-style ExpressionData.csv /
   PseudoTime.csv, the format `FlowGRN/get_traj_real_dropout.py` expects.
3. `run_trajectory_reconstruction` — subprocess call per branch into the
   `flowgrn` conda env (torch/torchcfm live there, not in this kernel's env).
4. `concatenate_trajectories` — stitch per-branch traj.npy into one
   (n_time, n_cell, n_gene) array, as FlowGRN's own README instructs.
5. `run_dyngenie3` — call dynGENIE3 directly (pure sklearn/numpy, no torch, so
   this runs in-process) restricted to a given candidate-regulator gene list.
6. `threshold_vim_topk` — keep only the top-K regulators per target gene.
7. `edges_to_signed_graph` — sparse edge list -> networkx graph, sign looked up
   from a curated db (falling back to sign=0 — bonesis lets the solver decide
   the monotonicity for edges FlowGRN discovered outside the curated database).
"""

import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from grn_inference.grn import df_to_graph
from grn_inference.tf_database import lookup_sign_from_db


def derive_branches(adata) -> dict:
    """One branch per terminal state: the `label`s on the path from
    `adata.uns['root_state']` to that terminal, per `adata.uns['trajectory_edges']`.

    Returns {terminal_state: [label, ...]} (root first, terminal last).
    """
    root = adata.uns["root_state"]
    terminals = list(adata.uns["terminal_states"])
    tree = nx.DiGraph()
    tree.add_edges_from(adata.uns["trajectory_edges"])

    return {terminal: nx.shortest_path(tree, root, terminal) for terminal in terminals}


def write_beeline_inputs(adata, expr: pd.DataFrame, branches: dict, out_dir) -> dict:
    """Write one combined ExpressionData.csv (genes x cells) and PseudoTime.csv
    (cells x branch, NaN outside membership) covering the union of cells across
    all branches. Column order of PseudoTime.csv defines FlowGRN's --branch-id.

    `expr` must be a cells x genes DataFrame (same orientation as `data` in
    "5. GRN Construction.ipynb"); it is written out transposed (genes x cells)
    because that's the orientation get_traj_real_dropout.py expects.

    Returns {"expr_fn", "pseudoT_fn", "branch_order"} (branch_order[i] is the
    terminal-state name corresponding to --branch-id i).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = adata.obs["label"].astype(str)
    pseudotime = adata.obs["pseudotime"].astype(float)

    branch_order = list(branches.keys())
    membership = {
        terminal: pseudotime.where(label.isin(path_labels))
        for terminal, path_labels in branches.items()
    }
    time_df = pd.DataFrame(membership)[branch_order].dropna(how="all")

    cells = time_df.index.intersection(expr.index)
    time_df = time_df.loc[cells]

    expr_fn = out_dir / "ExpressionData.csv"
    pseudoT_fn = out_dir / "PseudoTime.csv"

    expr.loc[cells].T.to_csv(expr_fn)
    time_df.to_csv(pseudoT_fn)

    return {"expr_fn": str(expr_fn), "pseudoT_fn": str(pseudoT_fn), "branch_order": branch_order}


def run_trajectory_reconstruction(
    expr_fn,
    pseudoT_fn,
    branch_id: int,
    out_dir,
    *,
    flowgrn_repo,
    flowgrn_python,
    seed: int = 0,
    n_snapshot: int = 5,
    extra_args=None,
) -> Path:
    """Run FlowGRN's get_traj_real_dropout.py for one branch, via subprocess in
    the `flowgrn` conda env. Returns the path to the produced traj.npy
    (get_traj_real_dropout.py saves it at `{out_dir}/{branch_id}/traj.npy`).
    """
    flowgrn_repo = Path(flowgrn_repo)
    cmd = [
        str(flowgrn_python),
        str(flowgrn_repo / "get_traj_real_dropout.py"),
        "--expr-fn", str(expr_fn),
        "--pseudoT-fn", str(pseudoT_fn),
        "--out-dir", str(out_dir),
        "--seed", str(seed),
        "--branch-id", str(branch_id),
        "--n-snapshot", str(n_snapshot),
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True, cwd=flowgrn_repo)
    return Path(out_dir) / str(branch_id) / "traj.npy"


def concatenate_trajectories(traj_paths) -> np.ndarray:
    """Stitch per-branch traj.npy arrays (n_time, n_cell_branch, n_gene) into one
    (n_time, total_n_cell, n_gene) array, as FlowGRN's README instructs before
    running network inference. Assumes all branches share n_time/n_gene (i.e.
    the same --n-snapshot and the same shared ExpressionData.csv)."""
    trajs = [np.load(p) for p in traj_paths]
    return np.concatenate(trajs, axis=1)


def run_dyngenie3(
    traj: np.ndarray,
    gene_names: list,
    regulators,
    *,
    flowgrn_repo,
    ntrees: int = 1000,
    nthreads: int = 1,
) -> pd.DataFrame:
    """Run dynGENIE3 directly on a concatenated trajectory array
    (n_time, n_cell, n_gene), restricted to `regulators` as candidate TFs.

    dynGENIE3 is pure sklearn/numpy (no torch), so this runs in-process even
    outside the `flowgrn` env — only trajectory reconstruction needs that env.

    Returns the VIM as a DataFrame (VIM.loc[i, j] = importance of i as a
    regulator of j), indexed/columned by `gene_names`.
    """
    flowgrn_repo = str(Path(flowgrn_repo))
    if flowgrn_repo not in sys.path:
        sys.path.insert(0, flowgrn_repo)
    from dynGENIE3 import dynGENIE3

    n_time, n_cell, n_gene = traj.shape
    timestep = [np.arange(n_time) for _ in range(n_cell)]
    TS_data = [traj.transpose(1, 0, 2)[i] for i in range(n_cell)]

    VIM, alphas, prediction_score, stability_score, _ = dynGENIE3(
        TS_data,
        timestep,
        gene_names=gene_names,
        regulators=regulators,
        compute_quality_scores=True,
        ntrees=ntrees,
        nthreads=nthreads,
    )
    return pd.DataFrame(VIM, index=gene_names, columns=gene_names)


def threshold_vim_topk(vim: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Keep only the top-`k` regulators per target gene (column), by VIM weight.

    Returns a long-format edge list with columns [source, target, weight].
    """
    edges = []
    for target in vim.columns:
        col = vim[target].drop(index=target, errors="ignore")
        for source, weight in col.nlargest(k).items():
            if weight > 0:
                edges.append((source, target, weight))
    return pd.DataFrame(edges, columns=["source", "target", "weight"])


def edges_to_signed_graph(edges: pd.DataFrame, sign_db: pd.DataFrame) -> nx.MultiDiGraph:
    """Attach a curated sign (via `lookup_sign_from_db`) to each FlowGRN-inferred
    edge, falling back to sign=0 ("either monotonicity" — bonesis lets the
    solver pick the role from the data) for pairs absent from `sign_db`.
    """
    edges = edges.copy()
    edges["sign"] = [
        lookup_sign_from_db(row.source, row.target, sign_db) for row in edges.itertuples()
    ]
    return df_to_graph(edges, source="source", target="target")
