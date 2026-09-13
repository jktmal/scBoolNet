#!/usr/bin/env python
"""Dynamo stochastic — CBDir: cycle_benchmarking_dynamo.ipynb"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rna_velocity.cli import load_adata, parse_io_args, save_adata


def main():
    args = parse_io_args("dynamo")
    import dynamo as dyn

    adata = load_adata(args.input)
    dyn.pp.recipe_monocle(adata, n_top_genes=1000, fg_kwargs={"shared_count": 20})
    dyn.tl.dynamics(adata, model="stochastic")
    dyn.tl.neighbors(adata, n_neighbors=30)
    dyn.tl.reduceDimension(adata, n_pca_components=30)
    dyn.tl.cell_velocities(adata, method="pearson", other_kernels_dict={"transform": "sqrt"})
    if "velocity_S" not in adata.layers and "velocity_umap" in adata.obsm:
        pass  # projected only
    save_adata(adata, args.output, vkey="velocity_S", method="dynamo")


if __name__ == "__main__":
    main()
