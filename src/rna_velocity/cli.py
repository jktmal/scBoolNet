"""Minimal shared CLI I/O for velocity runners."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional


def parse_io_args(method: str, extra: Optional[Dict[str, Dict[str, Any]]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Run RNA velocity method: {method}")
    p.add_argument("--input", required=True, type=Path, help="Input .h5ad (spliced/unspliced)")
    p.add_argument("--output", required=True, type=Path, help="Output .h5ad with layers['velocity']")
    p.add_argument("--cluster-key", default="clusters", help="obs column for cell types/clusters")
    for name, kwargs in (extra or {}).items():
        p.add_argument(name, **kwargs)
    return p.parse_args()


def load_adata(path: Path):
    import scanpy as sc

    return sc.read_h5ad(path)


def save_adata(adata, path: Path, *, vkey: str = "velocity", method: Optional[str] = None):
    """Write AnnData; ensure layers['velocity'] exists (alias from native vkey)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if vkey != "velocity" and vkey in adata.layers and "velocity" not in adata.layers:
        adata.layers["velocity"] = adata.layers[vkey]

    adata.uns["velocity_vkey"] = vkey
    if method is not None:
        adata.uns["velocity_method"] = method
    adata.write_h5ad(path)
    print(f"Wrote {path} (vkey={vkey})")
