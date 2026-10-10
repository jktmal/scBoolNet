"""Cell-cycle phase scoring via scran::cyclone(), called from R through rpy2.

Same algorithm/call as scHSC_herault/R_src/data_preparation.R::getCellCyclePhases
(Scialdone et al. gene-pair method) — used to reproduce the cell-cycle covariates
(G1_score/S_score/G2M_score) regressed out in the Seurat integration step
(scHSC_herault/R_src/Seurat3_integration.R:148).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

# rpy2 must find the R of the active conda env (not any system R on PATH) —
# R_HOME has to be set before the first `import rpy2...`, so this runs at
# module import time, before pulling in rpy2 itself.
os.environ.setdefault("R_HOME", str(Path(sys.prefix) / "lib" / "R"))

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402


def score_cell_cycle_with_scran_cyclone(
    counts: "np.ndarray | sp.spmatrix",
    gene_ids: list[str],
    cell_ids: list[str],
    n_workers: int = 1,
) -> pd.DataFrame:
    """scran::cyclone() (Scialdone et al. pairs method) via rpy2.

    `counts` must be raw (unnormalized) UMI counts, cells x genes, in the same
    gene order as `gene_ids`; `gene_ids` must be Ensembl IDs (mouse_cycle_markers.rds
    is keyed on Ensembl).
    """
    importr("Matrix")
    importr("scran")
    importr("BiocParallel")

    with tempfile.TemporaryDirectory() as tmpdir:
        mtx_path = Path(tmpdir) / "counts.mtx"
        sio.mmwrite(mtx_path, sp.csr_matrix(counts).T)  # genes x cells, like exprs(gbm_cds) in R
        ro.r(f'm <- as(Matrix::readMM("{mtx_path}"), "CsparseMatrix")')

    ro.globalenv["gene_ids"] = ro.StrVector(gene_ids)
    ro.globalenv["cell_ids"] = ro.StrVector(cell_ids)
    ro.r("rownames(m) <- gene_ids; colnames(m) <- cell_ids")

    ro.r("set.seed(100)")  # data_preparation.R:7
    ro.r('mm.pairs <- readRDS(system.file("exdata", "mouse_cycle_markers.rds", package="scran"))')
    ro.r(
        f"assignments <- cyclone(m, mm.pairs, gene.names = rownames(m), "
        f"BPPARAM = MulticoreParam({n_workers}, RNGseed = 100))"
    )

    scores = pd.DataFrame(
        {
            "G1_score": np.array(ro.r("assignments$scores$G1")),
            "S_score": np.array(ro.r("assignments$scores$S")),
            "G2M_score": np.array(ro.r("assignments$scores$G2M")),
        },
        index=cell_ids,
    )
    scores["phase"] = list(ro.r("as.character(assignments$phases)"))
    # data_preparation.R:16-17 renames cyclone's G1/G2M labels to the covariate
    # names used later in the regression step (Seurat3_integration.R:148:
    # "G2M_score+S_score+G1_score")
    scores["phase"] = scores["phase"].replace({"G1": "G1_G0", "G2M": "G2_M"})
    return scores
