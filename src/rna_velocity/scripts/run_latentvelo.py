#!/usr/bin/env python
"""LatentVelo — CBDir: cycle_benchmarking_latentvelo.ipynb (env: latentvelo)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from rna_velocity.cli import load_adata, parse_io_args, save_adata


def main():
    args = parse_io_args("latentvelo", extra={
            "--seed": {"type": int, "default": 521},
            "--epochs": {"type": int, "default": 24},
            "--workdir": {"type": Path, "default": None, "help": "chdir before train (optional)"},
        })
    import latentvelo as ltv
    import scvelo as scv

    if args.workdir:
        import os

        os.chdir(args.workdir)

    torch.manual_seed(args.seed)
    adata = load_adata(args.input)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    adata = ltv.utils.anvi_clean_recipe(adata, celltype_key=args.cluster_key)
    model = ltv.models.AnnotVAE(
        observed=2000,
        latent_dim=30,
        encoder_hidden=35,
        zr_dim=2,
        h_dim=3,
        celltypes=len(adata.obs[args.cluster_key].unique()),
    )
    ltv.train(
        model,
        adata,
        batch_size=100,
        learning_rate=1e-2,
        epochs=args.epochs,
        name=f"{args.output.stem}_parameters",
        grad_clip=100,
    )
    latent_adata, adata = ltv.output_results(model, adata, gene_velocity=True, decoded=True)
    # CBDir evaluates spliced_velocity on latent_adata
    if "spliced_velocity" in latent_adata.layers:
        out = latent_adata
        vkey = "spliced_velocity"
    else:
        out = adata
        vkey = "spliced_velocity" if "spliced_velocity" in adata.layers else "velocity"
    save_adata(out, args.output, vkey=vkey, method="latentvelo")


if __name__ == "__main__":
    main()
