#!/usr/bin/env python
"""DeepVelo — CBDir: cycle_benchmarking_DeepVelo2024.ipynb (env: deepvelo)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rna_velocity.cli import load_adata, parse_io_args, save_adata


def main():
    args = parse_io_args("deepvelo")
    import deepvelo as dv
    import scvelo as scv
    from deepvelo.utils import update_dict
    from deepvelo.utils.preprocess import autoset_coeff_s
    from deepvelo.utils.util import Constants

    adata = load_adata(args.input)
    configs = {
        "name": "DeepVelo",
        "loss": {"args": {"coeff_s": autoset_coeff_s(adata)}},
        "trainer": {"verbosity": 0},
    }
    configs = update_dict(Constants.default_configs, configs)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=3000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    dv.train(adata, configs)  # notebook sometimes passes Constants.default_configs; use merged configs
    save_adata(adata, args.output, vkey="velocity", method="deepvelo")


if __name__ == "__main__":
    main()
