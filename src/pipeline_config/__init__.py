"""Dataset selection and file layout for the notebook pipeline (see `pipeline_config.toml`).

`load_config` needs only the standard library; `prepare` pulls in scanpy, so it is imported lazily.
"""

from pipeline_config.config import ARTIFACTS, PipelineConfig, load_config, sample_name

__all__ = ["ARTIFACTS", "PipelineConfig", "load_config", "sample_name", "ensure_corrected_h5ad"]


def __getattr__(name):
    if name == "ensure_corrected_h5ad":
        from pipeline_config.prepare import ensure_corrected_h5ad

        return ensure_corrected_h5ad
    raise AttributeError(name)
