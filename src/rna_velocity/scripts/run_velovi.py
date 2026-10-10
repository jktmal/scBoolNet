#!/usr/bin/env python
"""veloVI — CBDir: cycle_benchmarking_veloVI.ipynb"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from rna_velocity.cli import load_adata, parse_io_args, save_adata
from rna_velocity.compat import ensure_anndata_io

ensure_anndata_io()  # scvi-tools (via velovi) imports `anndata.io` — see rna_velocity/compat.py


def add_velovi_outputs_to_adata(adata, vae):
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")
    t = latent_time
    scaling = 20 / t.max(0)
    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time
    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr).detach().cpu().numpy()
    ) * scaling
    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0


def main():
    args = parse_io_args("velovi")
    import scanpy as sc
    import scvelo as scv
    from velovi import VELOVI, preprocess_data

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
    adata = preprocess_data(adata)
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train()
    add_velovi_outputs_to_adata(adata, vae)
    save_adata(adata, args.output, vkey="velocity", method="velovi")


if __name__ == "__main__":
    main()
