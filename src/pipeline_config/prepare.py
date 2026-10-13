"""Bridge between preprocessing and `3. Final Cell Annotation` for runs with or without integration.

`3.` only needs `X` + `obs_names` of `integrated_combat_corrected_expression.h5ad`. With several
batches that file comes from `2. Data Integration` (ComBat + cell-cycle regression + scale). With a
single dataset there is nothing to integrate, so `ensure_corrected_h5ad` builds the equivalent file
without the ComBat step (log-norm on the canonical HVG set -> optional cell-cycle regression -> scale).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

from pipeline_config.config import PipelineConfig

CELL_CYCLE_COLUMNS = ["G1_score", "S_score", "G2M_score"]


def regress_cycle_and_scale(adata, cell_cycle_scores_tsv: Path, *, max_value: float = 10) -> None:
    """Seurat `ScaleData(vars.to.regress=G1+S+G2M)` equivalent, in place (shared with `2. Data Integration`)."""
    cc_scores = pd.read_csv(cell_cycle_scores_tsv, sep="\t", index_col=0)
    missing = adata.obs_names.difference(cc_scores.index)
    assert missing.empty, f"{len(missing)} komórek bez wyniku cyclone w {cell_cycle_scores_tsv}"

    for col in CELL_CYCLE_COLUMNS:
        adata.obs[col] = cc_scores.loc[adata.obs_names, col].to_numpy()

    sc.pp.regress_out(adata, CELL_CYCLE_COLUMNS)
    sc.pp.scale(adata, max_value=max_value)


def build_unintegrated_corrected(cfg: PipelineConfig):
    """Single-dataset counterpart of the ComBat-corrected matrix produced by `2. Data Integration`."""
    if str(cfg.root) not in sys.path:
        sys.path.insert(0, str(cfg.root))
    from data_prep import normalize_log_scanpy  # repo-root module, same one notebook 2 uses

    adata = sc.read_h5ad(cfg.path("merged_h5ad"))
    hvg_genes = pd.read_csv(cfg.path("hvgs_csv"), index_col=0, nrows=0).columns
    missing = pd.Index(hvg_genes).difference(adata.var_names)
    assert missing.empty, f"{len(missing)} genów z {cfg.path('hvgs_csv')} brakuje w {cfg.path('merged_h5ad')}"

    adata_hvg = adata[:, hvg_genes].copy()
    normalize_log_scanpy(adata_hvg)
    adata_hvg.layers["lognorm"] = adata_hvg.X.copy()
    adata_hvg.obs["batch"] = adata_hvg.obs["batch"].astype("category")

    if cfg.regress_cell_cycle:
        regress_cycle_and_scale(adata_hvg, cfg.path("cell_cycle_scores_tsv"))
    else:
        sc.pp.scale(adata_hvg, max_value=10)
    return adata_hvg


def ensure_corrected_h5ad(cfg: PipelineConfig) -> Path:
    """Path of the corrected-expression h5ad that `3. Final Cell Annotation` reads.

    integrate=True  -> must already exist (written by `2. Data Integration`).
    integrate=False -> built here from `merged_dataset.h5ad` and written to the same path.
    """
    out = cfg.path("corrected_h5ad")
    if cfg.integrate:
        if not out.exists():
            raise FileNotFoundError(f"{out} not found — run '2. Data Integration.ipynb' first (integrate=true)")
        return out

    build_unintegrated_corrected(cfg).write_h5ad(out)
    return out
