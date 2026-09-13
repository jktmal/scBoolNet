#!/usr/bin/env python
"""pyro-velocity — CBDir: cycle_benchmarking_pyrovelocity.ipynb (env: pyrovelocity)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rna_velocity.cli import load_adata, parse_io_args, save_adata


def compute_mean_vector_field(pos, adata, basis="umap", n_jobs=1, spliced="spliced_pyro", raw=False):
    """From CBDir notebook (writes layers['velocity_pyro'])."""
    import scvelo as scv

    scv.pp.neighbors(adata, use_rep="pca")
    adata.var["velocity_genes"] = True

    if spliced == "spliced_pyro":
        ut, st = pos["ut"], pos["st"]
        if raw:
            ut = ut / ut.sum(axis=-1, keepdims=True)
            st = st / st.sum(axis=-1, keepdims=True)
        adata.layers["spliced_pyro"] = st.mean(0).squeeze()
        if "u_scale" in pos:
            adata.layers["velocity_pyro"] = (
                ut * pos["beta"] / pos["u_scale"] - st * pos["gamma"]
            ).mean(0)
        elif "beta_k" in pos:
            adata.layers["velocity_pyro"] = (
                (ut * pos["beta_k"] - pos["st"] * pos["gamma_k"]).mean(0).squeeze()
            )
        else:
            adata.layers["velocity_pyro"] = (ut * pos["beta"] - pos["st"] * pos["gamma"]).mean(0)
        scv.tl.velocity_graph(adata, vkey="velocity_pyro", xkey="spliced_pyro", n_jobs=n_jobs)
    scv.tl.velocity_embedding(adata, vkey="velocity_pyro", basis=basis)


def main():
    args = parse_io_args("pyrovelocity", extra={
            "--epochs": {"type": int, "default": 1000},
            "--batch-size": {"type": int, "default": 4000},
            "--gpu": {"type": int, "default": 0},
        })
    import scvelo as scv
    from pyrovelocity.api import train_model

    adata = load_adata(args.input)
    adata.layers["raw_spliced"] = adata.layers["spliced"]
    adata.layers["raw_unspliced"] = adata.layers["unspliced"]
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=3000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    adata.obs["u_lib_size_raw"] = adata.layers["raw_unspliced"].toarray().sum(-1)
    adata.obs["s_lib_size_raw"] = adata.layers["raw_spliced"].toarray().sum(-1)

    adata_model_pos = train_model(
        adata,
        max_epochs=args.epochs,
        svi_train=True,
        log_every=100,
        patient_init=45,
        batch_size=args.batch_size,
        use_gpu=args.gpu,
        cell_state="state_info",
        include_prior=True,
        offset=True,
        library_size=True,
        patient_improve=1e-3,
        model_type="auto",
        guide_type="auto",
        train_size=1.0,
    )
    compute_mean_vector_field(adata_model_pos[1], adata)
    if "velocity" not in adata.layers and "velocity_pyro" in adata.layers:
        adata.layers["velocity"] = adata.layers["velocity_pyro"]
    save_adata(adata, args.output, vkey="velocity_pyro", method="pyrovelocity")


if __name__ == "__main__":
    main()
