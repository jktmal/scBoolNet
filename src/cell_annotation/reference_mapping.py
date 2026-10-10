"""scVI/scArches-style reference mapping for automated preliminary cell annotation.

Ported from the tangle of ~280 near-duplicate cells in `1. Cell Annotation.ipynb`
(three copies of the same recipe: FACS reference, self-trained CELLxGENE census
reference, pretrained CELLxGENE census model). Unifies both entry points behind
the same gene-alignment / query-mapping / label-transfer functions:

- a pretrained reference model downloaded from CELLxGENE census
  (`load_pretrained_reference_model` + `align_query_to_model`), or
- a reference trained from scratch on raw reference counts
  (`train_reference_model`).

Both then go through the same `map_query_to_reference` -> `get_latent_adata` ->
`combine_reference_query` / `transfer_labels` pipeline.

Deliberately drops the `scarches` package: `scvi.model.SCVI.load_query_data` /
`prepare_query_anndata` already implement the same architecture-surgery
transfer learning natively, so the `sys.modules['anndata'] = ad` compat shim
the notebook needed for `scarches` is no longer necessary. Also drops the
manual zero-padding-and-concat gene alignment for the pretrained-model path
(`missing_gene_adata` in the old notebook) in favour of
`SCVI.prepare_query_anndata`, which already does this.

The FACS (GSE90742) reference path was abandoned in the notebook (marked
"DALEJ NIE IDZIEMY") and is not ported here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi


def ensure_batch_key(adata: ad.AnnData, *, batch_key: str = "batch", value: str = "0") -> None:
    """Set `obs[batch_key]` to a constant categorical value if not already present.

    scVI's `setup_anndata` requires a batch key even for a single-batch
    reference/query dataset.
    """
    if batch_key not in adata.obs:
        adata.obs[batch_key] = value
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")


def map_var_names_to_census_ids(
    adata: ad.AnnData,
    *,
    census_version: str,
    organism: str = "mus_musculus",
    drop_unmapped: bool = True,
) -> ad.AnnData:
    """Rename `adata.var_names` from gene symbols to CELLxGENE census Ensembl feature ids.

    Both the pretrained census scVI model and a reference trained on raw
    census data index genes by `feature_id` (Ensembl id), not gene symbol —
    our own data uses symbols, so this remapping is needed before either
    `align_query_to_pretrained_model` or training/using a census-based
    reference. Returns a copy; `adata` itself is left untouched.
    """
    import cellxgene_census

    with cellxgene_census.open_soma(census_version=census_version) as census:
        genes_df = pd.DataFrame(cellxgene_census.get_var(census, organism=organism))
    name_to_ensembl = dict(zip(genes_df["feature_name"], genes_df["feature_id"]))

    out = adata.copy()
    out.var["gene_symbol"] = out.var_names
    out.var_names = [name_to_ensembl.get(g, "Unknown") for g in out.var_names]
    if drop_unmapped:
        out = out[:, out.var_names != "Unknown"].copy()
    return out


def load_pretrained_reference_model(model_path: str | Path) -> scvi.model.SCVI:
    """Load a pretrained scVI reference model (e.g. downloaded from CELLxGENE census)."""
    return scvi.model.SCVI.load(str(model_path))


def align_query_to_model(query: ad.AnnData, reference_model: scvi.model.SCVI) -> ad.AnnData:
    """Reorder/pad `query` to the gene set `reference_model` expects.

    Thin wrapper around `SCVI.prepare_query_anndata`, which reorders to the
    model's registered feature set and zero-fills genes missing from `query`
    — the same thing the notebook did by hand with a `csr_matrix` of zeros +
    `sc.concat`. Works for any `scvi.model.SCVI` reference (pretrained or
    self-trained), not just a downloaded one. `query.var_names` must already
    be in the model's id space (Ensembl for CELLxGENE census models — see
    `map_var_names_to_census_ids`).
    """
    query = query.copy()
    scvi.model.SCVI.prepare_query_anndata(query, reference_model)
    return query


def train_reference_model(
    reference: ad.AnnData,
    *,
    batch_key: str = "batch",
    n_layers: int = 2,
    max_epochs: Optional[int] = None,
    save_path: Optional[str | Path] = None,
    **scvi_kwargs,
) -> scvi.model.SCVI:
    """Train a new scVI reference model from scratch on `reference` counts."""
    ensure_batch_key(reference, batch_key=batch_key)
    scvi.model.SCVI.setup_anndata(reference, batch_key=batch_key)
    model = scvi.model.SCVI(
        reference,
        n_layers=n_layers,
        encode_covariates=True,
        deeply_inject_covariates=False,
        use_layer_norm="both",
        use_batch_norm="none",
        **scvi_kwargs,
    )
    model.train(max_epochs=max_epochs) if max_epochs is not None else model.train()
    if save_path is not None:
        model.save(str(save_path), overwrite=True)
    return model


def plot_training_history(model: scvi.model.SCVI, *, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot `model`'s training (+ validation, if present) loss/ELBO curve.

    The old notebook did this by hand (`vae.history["train_loss"].plot()`)
    but always against a stale/undefined model variable, so the plot never
    actually ran. The exact history key name varies a bit across scvi-tools
    versions/model types, so this tries the common ones instead of hardcoding
    `"train_loss"`.
    """
    history = model.history
    train_key = next(
        (k for k in ("elbo_train", "train_loss_epoch", "train_loss") if k in history), None
    )
    if train_key is None:
        raise KeyError(f"no known training-loss key in model.history: {list(history)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    history[train_key].plot(ax=ax, label="train")
    val_key = train_key.replace("train", "validation")
    if val_key in history:
        history[val_key].plot(ax=ax, label="validation")
    ax.set_ylabel(train_key)
    ax.legend()
    return ax


def map_query_to_reference(
    query: ad.AnnData,
    reference_model: scvi.model.SCVI,
    *,
    max_epochs: int = 200,
    freeze_dropout: bool = True,
    save_path: Optional[str | Path] = None,
) -> scvi.model.SCVI:
    """Surgery: map `query` onto `reference_model`'s latent space via transfer learning."""
    query_model = scvi.model.SCVI.load_query_data(query, reference_model, freeze_dropout=freeze_dropout)
    query_model.train(max_epochs=max_epochs, plan_kwargs=dict(weight_decay=0.0))
    if save_path is not None:
        query_model.save(str(save_path), overwrite=True)
    return query_model


def get_latent_adata(
    model: scvi.model.SCVI,
    adata: ad.AnnData,
    *,
    role: str,
    cell_type_key: Optional[str] = None,
) -> ad.AnnData:
    """Wrap `model`'s latent representation of `adata` in its own AnnData for plotting/KNN.

    `role` is stamped into `obs["reference_or_query"]` (typically `"reference"`
    or `"query"`). `cell_type_key`, if given and present in `adata.obs`, is
    carried over as `obs["cell_type"]`; otherwise `cell_type` is set to `None`
    (the usual case for a query dataset awaiting label transfer).
    """
    latent = sc.AnnData(X=model.get_latent_representation(adata=adata), obs=adata.obs.copy())
    latent.obs["reference_or_query"] = role
    if cell_type_key is not None and cell_type_key in adata.obs:
        latent.obs["cell_type"] = adata.obs[cell_type_key].tolist()
    else:
        latent.obs["cell_type"] = None
    return latent


def combine_reference_query(reference_latent: ad.AnnData, query_latent: ad.AnnData) -> ad.AnnData:
    """Concatenate reference/query latent embeddings and compute a joint neighbors graph + UMAP.

    Raises if the combined embedding contains NaNs — `sc.pp.neighbors`/`sc.tl.umap`
    otherwise fail on them with a much more cryptic error. The old notebook
    checked for this by hand (`check_X.isna().sum()`) but never acted on it.
    """
    combined = sc.concat(
        [reference_latent, query_latent],
        axis=0,
        join="outer",
        index_unique=None,
        merge="unique",
    )
    n_nan = int(np.isnan(combined.X).sum())
    if n_nan:
        raise ValueError(
            f"{n_nan} NaN values in the combined reference+query latent embedding "
            "— check that reference and query were aligned to the same model/genes."
        )
    sc.pp.neighbors(combined)
    sc.tl.umap(combined)
    return combined


def transfer_labels(
    reference_latent: ad.AnnData,
    query_latent: ad.AnnData,
    *,
    cell_type_key: str = "cell_type",
    n_neighbors: int = 30,
) -> pd.Series:
    """KNN-majority-vote label transfer in latent space: the step the old notebook never reached.

    Fits a KNN classifier on the reference latent embedding + its known
    `cell_type_key` labels, and predicts a label for every query cell from its
    nearest reference neighbours. Returns a `pd.Series` indexed by
    `query_latent.obs_names`.
    """
    from sklearn.neighbors import KNeighborsClassifier

    ref_labels = reference_latent.obs[cell_type_key]
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(reference_latent.X, ref_labels)
    predicted = knn.predict(query_latent.X)
    return pd.Series(predicted, index=query_latent.obs_names, name=cell_type_key)
