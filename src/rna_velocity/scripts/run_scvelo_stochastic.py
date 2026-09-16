#!/usr/bin/env python
"""scVelo stochastic — CBDir: cycle_benchmark_scVelo_stochastic.ipynb"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rna_velocity.cli import load_adata, parse_io_args, save_adata


def main():
    args = parse_io_args("scvelo_stochastic")
    import scvelo as scv

    import scanpy as sc

    adata = load_adata(args.input)
    # scvelo 0.3.x filter_and_normalize() no longer does HVG selection via
    # n_top_genes (silently forwarded to normalize_per_cell(), which raises
    # TypeError) — select HVGs explicitly instead.
    scv.pp.filter_genes(adata, min_shared_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable].copy()
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, vkey="velocity", mode="stochastic")
    save_adata(adata, args.output, vkey="velocity", method="scvelo_stochastic")


if __name__ == "__main__":
    main()
