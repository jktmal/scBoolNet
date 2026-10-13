"""Shared trajectory-inference contract: TI (PAGA+DPT) and RNA Velocity (CellRank)
notebooks both fill `obs['pseudotime']`/`obs['branch']`/`uns['root_state'/'terminal_states'/
'transient_states']` and use this package for the export/plotting steps that follow."""

from trajectory_inference.export import (
    apply_bonesis_constraints,
    bonesis_spec_from_tree,
    derive_bonesis_constraints,
    render_bonesis_lines,
    select_observations,
    write_bonesis_constraints,
    write_trajectory_h5ad,
)
from trajectory_inference.plotting import (
    plot_marker_genes,
    plot_pseudotime_boxplot,
    plot_pseudotime_comparison,
    plot_pseudotime_kde,
    plot_state_graph,
)

__all__ = [
    "select_observations",
    "derive_bonesis_constraints",
    "bonesis_spec_from_tree",
    "render_bonesis_lines",
    "apply_bonesis_constraints",
    "write_bonesis_constraints",
    "write_trajectory_h5ad",
    "plot_pseudotime_boxplot",
    "plot_pseudotime_kde",
    "plot_state_graph",
    "plot_marker_genes",
    "plot_pseudotime_comparison",
]
