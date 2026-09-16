"""Compat shims for third-party version skew inside the `velo-core` env.

scvi-tools (a veloVI dependency) moved some anndata re-exports into two
submodules — `anndata.io` (`read_csv`/`read_loom`/`read_text`/`read_elem`)
and `anndata.abc` (`CSCDataset`/`CSRDataset`) — that only exist in
anndata>=0.11. dynamo-release pins `anndata<0.11.0`, and we'd rather keep
one shared anndata version across `velo-core` than break dynamo, so this
module stubs both submodules from what anndata 0.10.x already exposes
(top-level + `anndata.experimental`) instead of upgrading anndata.

Alternative considered: pin an older, pre-anndata.io scvi-tools (<1.2.2)
instead of shimming — no code needed, but it drags in jax/jaxlib/flax as a
~70MB base dependency that VELOVI/VELOVAE (pure PyTorch) never use. Verified
both work once the real libomp/segfault issue (see
envs/rna_velocity/fix_macos_libomp.sh) was fixed at its actual source rather
than worked around here. Kept the shim over the extra ~15 packages, but it's
a judgment call, not a hard requirement — see docs/chat-notes-veloBench.md
§14 pkt 7 for the full comparison.
"""
from __future__ import annotations

import sys
import types


def ensure_anndata_io() -> None:
    """Stub `anndata.io` / `anndata.abc` if the installed anndata lacks them.

    Call before importing anything that needs them (currently: scvi-tools,
    via velovi). No-op once anndata itself provides these (>=0.11).
    """
    import anndata

    try:
        import anndata.io  # noqa: F401
    except ImportError:
        import anndata.experimental as ad_experimental

        io_shim = types.ModuleType("anndata.io")
        io_shim.read_csv = anndata.read_csv
        io_shim.read_loom = anndata.read_loom
        io_shim.read_text = anndata.read_text
        io_shim.read_elem = ad_experimental.read_elem
        sys.modules["anndata.io"] = io_shim
        anndata.io = io_shim

    try:
        import anndata.abc  # noqa: F401
    except ImportError:
        import anndata.experimental as ad_experimental

        abc_shim = types.ModuleType("anndata.abc")
        abc_shim.CSCDataset = ad_experimental.CSCDataset
        abc_shim.CSRDataset = ad_experimental.CSRDataset
        sys.modules["anndata.abc"] = abc_shim
        anndata.abc = abc_shim
