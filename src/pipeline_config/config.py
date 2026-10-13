"""Single source of truth for which dataset the pipeline runs on and where its files live.

Only the standard library is used, so every notebook kernel (scBoolSeq, scrna-ti, velo-core)
can import it. See `pipeline_config.toml` for the knobs.
"""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "pipeline_config.toml"
DATASET_ENV_VAR = "BOOLAI_DATASET"
CONFIG_ENV_VAR = "BOOLAI_CONFIG"

# artifact name -> filename inside results/<dataset>/
ARTIFACTS = {
    "merged_h5ad": "merged_dataset.h5ad",
    "merged_csv": "merged_dataset.csv",
    "counts_hvgs_csv": "counts_hvgs.csv",
    "hvgs_csv": "dataset_hvgs.csv",
    "preliminary_annotation_tsv": "preliminary_cell_annotation_metadata.tsv",
    "cell_cycle_scores_tsv": "cell_cycle_scores.tsv",
    "integration_metrics_csv": "integration_metrics.csv",
    "integration_summary_json": "integration_run_summary.json",
    "corrected_h5ad": "integrated_combat_corrected_expression.h5ad",
    "binarizer_joblib": "scBoolSeqBinarizer_fitted.joblib",
    "scboolseq_obs_csv": "scboolseq_inferred_observations.csv",
    "cell_annotation_tsv": "cell_annotation_metadata.tsv",
}

_SAMPLE_PREFIX = re.compile(r"^GSM\d+_")
_SAMPLE_SUFFIX = re.compile(r"_merged$")


def sample_name(sample_file: str) -> str:
    """`GSM4443875_young_A_merged.tsv` -> `young_A` (the `batch` label)."""
    stem = Path(sample_file).name.split(".")[0]
    return _SAMPLE_SUFFIX.sub("", _SAMPLE_PREFIX.sub("", stem))


@dataclass(frozen=True)
class PipelineConfig:
    dataset: str
    samples: tuple[str, ...]
    integrate: bool
    preliminary_annotation: bool
    regress_cell_cycle: bool
    genes_file: str | None
    paper_metadata: str | None
    method: str
    aggregation: str
    organism: str
    markers_file: str | None
    root_candidates: tuple[str, ...]
    terminal_states: tuple[str, ...]
    transient_states: tuple[str, ...]
    root: Path = field(default=REPO_ROOT)

    # --- inputs -----------------------------------------------------------------
    @property
    def sample_paths(self) -> list[Path]:
        return [self.root / s for s in self.samples]

    @property
    def sample_names(self) -> list[str]:
        """Batch label per sample, in the same order as `samples`."""
        return [sample_name(s) for s in self.samples]

    @property
    def genes_path(self) -> Path | None:
        return self.root / self.genes_file if self.genes_file else None

    @property
    def markers_path(self) -> Path | None:
        """Markers file for this dataset: a .py module like `markers.py` (`states_and_markers`, `positive_markers`)."""
        return self.root / self.markers_file if self.markers_file else None

    def load_markers_module(self):
        """Import the configured markers file as a module — the same contract as `from markers import ...`."""
        if self.markers_path is None:
            raise ValueError(f"dataset {self.dataset!r}: set `markers_file` in pipeline_config.toml")
        if not self.markers_path.is_file():
            raise FileNotFoundError(f"markers file not found: {self.markers_path}")
        spec = importlib.util.spec_from_file_location(f"_markers_{self.markers_path.stem}", self.markers_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @property
    def paper_metadata_path(self) -> Path | None:
        return self.root / self.paper_metadata if self.paper_metadata else None

    # --- outputs ----------------------------------------------------------------
    @property
    def results_root(self) -> Path:
        """results/<dataset>/ — everything produced for this dataset."""
        return self.root / "results" / self.dataset

    def path(self, artifact: str) -> Path:
        """Path of a named artifact (see ARTIFACTS); parent directory is created."""
        if artifact not in ARTIFACTS:
            raise KeyError(f"unknown artifact {artifact!r}; choose from {sorted(ARTIFACTS)}")
        p = self.results_root / ARTIFACTS[artifact]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def integrated_path(self, integration_method: str) -> Path:
        """integrated_<scanorama|harmony>.h5ad from 2. Data Integration."""
        p = self.results_root / f"integrated_{integration_method}.h5ad"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def results_dir(self, method: str | None = None) -> Path:
        """results/<dataset>/<method>/ — trajectory.h5ad, transition_constraints.json, ..."""
        p = self.results_root / (method or self.method)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def grn_dir(self) -> Path:
        """results/<dataset>/grn/ — 4.5 / 5 / 6 file bundle."""
        p = self.results_root / "grn"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def file_prefix(self) -> str:
        """Prefix of the 4.5 / 5 / 6 file bundle (`<prefix>_pkn.json`, `<prefix>_config.json`, ...).

        Contains the directory, so use it only as `f"{file_prefix}_<name>"` (as `export_pkn_bundle` does).
        """
        return str((self.grn_dir / self.aggregation).relative_to(self.root))

    @property
    def meta_path(self) -> Path:
        """Per-phenotype binarized observations written by 4.5, read by 5 (and FlowGRN)."""
        return self.grn_dir / f"meta_{self.aggregation}_bin_aggregated.csv"

    def describe(self) -> str:
        steps = "integration" if self.integrate else "single dataset (no 1. Cell Annotation / 2. Data Integration)"
        return f"dataset={self.dataset!r} [{steps}], samples={list(self.sample_names)}, method={self.method!r}"


def _as_tuple(value) -> tuple[str, ...]:
    return tuple(value)


def load_config(dataset: str | None = None, config_path: str | Path | None = None) -> PipelineConfig:
    """Resolve the active dataset: explicit arg > $BOOLAI_DATASET > `active` in the config file."""
    config_path = Path(config_path or os.environ.get(CONFIG_ENV_VAR) or CONFIG_FILE)
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    name = dataset or os.environ.get(DATASET_ENV_VAR) or raw.get("active")
    datasets = raw.get("datasets", {})
    if name not in datasets:
        raise KeyError(f"dataset {name!r} not defined in {config_path}; defined: {sorted(datasets)}")

    merged = {**raw.get("defaults", {}), **datasets[name]}
    samples = _as_tuple(merged.get("samples", ()))
    if not samples:
        raise ValueError(f"dataset {name!r}: `samples` must list at least one input file")

    integrate = bool(merged.get("integrate", len(samples) > 1))
    if integrate and len(samples) < 2:
        raise ValueError(f"dataset {name!r}: integrate=true needs at least 2 samples, got {len(samples)}")

    genes_file = merged.get("genes_file") or None
    regress_cell_cycle = bool(merged.get("regress_cell_cycle", genes_file is not None))
    if regress_cell_cycle and not genes_file:
        raise ValueError(f"dataset {name!r}: regress_cell_cycle=true needs `genes_file` (Ensembl IDs)")

    return PipelineConfig(
        dataset=name,
        samples=samples,
        integrate=integrate,
        preliminary_annotation=bool(merged.get("preliminary_annotation", integrate)),
        regress_cell_cycle=regress_cell_cycle,
        genes_file=genes_file,
        paper_metadata=merged.get("paper_metadata") or None,
        method=merged.get("method", "paga_dpt"),
        aggregation=merged.get("aggregation", "sum"),
        organism=merged.get("organism", "mouse"),
        markers_file=merged.get("markers_file") or None,
        root_candidates=_as_tuple(merged.get("root_candidates", ())),
        terminal_states=_as_tuple(merged.get("terminal_states", ())),
        transient_states=_as_tuple(merged.get("transient_states", ())),
        root=config_path.parent,
    )
