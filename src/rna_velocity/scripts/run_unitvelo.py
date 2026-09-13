#!/usr/bin/env python
"""UniTVelo — CBDir: cycle_benchmarking_UniTVelo.ipynb (env: unitvelo)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rna_velocity.cli import load_adata, parse_io_args, save_adata


def main():
    args = parse_io_args("unitvelo", extra={"--gpu": {"type": int, "default": 0}})
    import scvelo as scv
    import unitvelo as utv

    adata = load_adata(args.input)
    adata.var["highly_variable"] = True
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=3000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    velo = utv.config.Configuration()
    velo.N_TOP_GENES = 3000
    velo.R2_ADJUST = True
    velo.IROOT = None
    velo.FIT_OPTION = "1"
    velo.GPU = args.gpu
    velo.vgenes = "offset"
    scv.pp.neighbors(adata)
    adata = utv.run_model(adata, args.cluster_key, config_file=velo)
    save_adata(adata, args.output, vkey="velocity", method="unitvelo")


if __name__ == "__main__":
    main()
