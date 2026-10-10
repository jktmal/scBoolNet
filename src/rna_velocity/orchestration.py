"""Minimal job launcher for RNA velocity CLIs across conda envs.

Notebook / caller (usually kernel ``velo-core``)::

    from rna_velocity.orchestration import launch, wait, list_outputs

    jobs = [
        launch("scvelo_stochastic", input_h5ad, output_h5ad),
        launch("deepvelo", input_h5ad, out_deep, background=True),
    ]
    wait(jobs)
    print(list_outputs("results", dataset="MyData"))
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from rna_velocity.registry import METHOD_ENV

# repo root = parents: rna_velocity -> src -> repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

METHOD_SCRIPT = {
    "scvelo_stochastic": "run_scvelo_stochastic.py",
    "dynamo": "run_dynamo.py",
    "velovi": "run_velovi.py",
    "deepvelo": "run_deepvelo.py",
    "unitvelo": "run_unitvelo.py",
    "latentvelo": "run_latentvelo.py",
    "pyrovelocity": "run_pyrovelocity.py",
}

PathLike = Union[str, Path]


@dataclass
class Job:
    method: str
    env: str
    cmd: List[str]
    output: Path
    log_path: Path
    proc: Optional[subprocess.Popen] = None
    returncode: Optional[int] = None
    extra: Dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        if self.returncode is not None:
            return True
        if self.proc is None:
            return False
        code = self.proc.poll()
        if code is not None:
            self.returncode = code
            return True
        return False

    @property
    def ok(self) -> bool:
        return self.done and self.returncode == 0


def default_output(results_dir: PathLike, method: str, dataset: str) -> Path:
    """``results/{method}/{dataset}_velocity.h5ad``."""
    return Path(results_dir) / method / f"{dataset}_velocity.h5ad"


def _script_path(method: str) -> Path:
    if method not in METHOD_SCRIPT:
        raise KeyError(f"Unknown method {method!r}. Known: {sorted(METHOD_SCRIPT)}")
    path = _SCRIPTS_DIR / METHOD_SCRIPT[method]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _build_cmd(
    method: str,
    input_path: Path,
    output_path: Path,
    *,
    cluster_key: str,
    conda_run: bool,
    extra_args: Sequence[str],
) -> List[str]:
    env = METHOD_ENV[method]
    if env is None:
        raise ValueError(f"Method {method!r} has no conda env in METHOD_ENV")

    script = _script_path(method)
    py_args = [
        "python",
        str(script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--cluster-key",
        cluster_key,
        *list(extra_args),
    ]
    # Ensure ``import rna_velocity`` works in the child process
    # (conda run inherits env; we set PYTHONPATH explicitly below).
    if conda_run:
        return ["conda", "run", "-n", env, "--no-capture-output", *py_args]
    return py_args


def launch(
    method: str,
    input_path: PathLike,
    output_path: PathLike,
    *,
    cluster_key: str = "clusters",
    background: bool = True,
    conda_run: bool = True,
    runs_dir: PathLike = "runs",
    extra_args: Optional[Sequence[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
) -> Job:
    """Start ``run_<method>.py`` in the mapped conda env.

    Parameters
    ----------
    background
        If True, return immediately (``Popen``). If False, block until exit.
    conda_run
        Wrap with ``conda run -n <env>``. Set False when already inside that env.
    """
    if method not in METHOD_ENV:
        raise KeyError(f"Unknown method {method!r}. Known: {sorted(METHOD_ENV)}")

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    log_path = runs / f"{method}_{output_path.stem}.log"

    cmd = _build_cmd(
        method,
        input_path,
        output_path,
        cluster_key=cluster_key,
        conda_run=conda_run,
        extra_args=extra_args or (),
    )

    child_env = os.environ.copy()
    src = str(_REPO_ROOT / "src")
    child_env["PYTHONPATH"] = src + os.pathsep + child_env.get("PYTHONPATH", "")
    if env_vars:
        child_env.update(env_vars)

    job = Job(
        method=method,
        env=METHOD_ENV[method],
        cmd=cmd,
        output=output_path,
        log_path=log_path,
    )

    log_f = open(log_path, "w")
    log_f.write(" ".join(cmd) + "\n\n")
    log_f.flush()

    if background:
        job.proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=child_env,
            cwd=str(_REPO_ROOT),
        )
        job.extra["pid"] = job.proc.pid
    else:
        completed = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=child_env,
            cwd=str(_REPO_ROOT),
        )
        job.returncode = completed.returncode
        log_f.close()
        if completed.returncode != 0:
            raise RuntimeError(
                f"{method} failed (exit {completed.returncode}); see {log_path}"
            )
    return job


def wait(
    jobs: Iterable[Job],
    *,
    poll_s: float = 5.0,
    raise_on_error: bool = True,
) -> List[Job]:
    """Block until all background jobs finish; return the same job list."""
    job_list = list(jobs)
    pending = [j for j in job_list if not j.done]
    while pending:
        time.sleep(poll_s)
        pending = [j for j in job_list if not j.done]

    # close log handles if still open via proc
    for j in job_list:
        if j.proc is not None and j.proc.stdout:
            try:
                j.proc.stdout.close()
            except Exception:
                pass

    failed = [j for j in job_list if not j.ok]
    if failed and raise_on_error:
        msgs = [f"{j.method} exit={j.returncode} log={j.log_path}" for j in failed]
        raise RuntimeError("Jobs failed:\n  " + "\n  ".join(msgs))
    return job_list


def list_outputs(
    results_dir: PathLike,
    *,
    dataset: Optional[str] = None,
    suffix: str = "_velocity.h5ad",
) -> Dict[str, Path]:
    """Map ``method -> path`` for existing velocity h5ads under ``results_dir``."""
    root = Path(results_dir)
    found: Dict[str, Path] = {}
    if not root.is_dir():
        return found

    for method in METHOD_ENV:
        method_dir = root / method
        if not method_dir.is_dir():
            continue
        if dataset is not None:
            candidate = method_dir / f"{dataset}{suffix}"
            if candidate.is_file():
                found[method] = candidate
            continue
        matches = sorted(method_dir.glob(f"*{suffix}"))
        if matches:
            found[method] = matches[-1]
    return found


def status(jobs: Iterable[Job]) -> List[Dict]:
    """Snapshot for notebook display."""
    rows = []
    for j in jobs:
        _ = j.done  # refresh poll
        rows.append(
            {
                "method": j.method,
                "env": j.env,
                "done": j.done,
                "ok": j.ok if j.done else None,
                "returncode": j.returncode,
                "output": str(j.output),
                "log": str(j.log_path),
                "pid": j.extra.get("pid"),
            }
        )
    return rows
