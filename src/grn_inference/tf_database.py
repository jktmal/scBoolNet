"""Loading and querying curated TF-target databases (CollecTRI)."""

import functools as fn
import os

import numpy as np
import pandas as pd


def load_collectri(organism: str = "mouse", cache_path: str = "collectri_mouse.csv") -> pd.DataFrame:
    """Load CollecTRI, downloading via decoupler and caching to `cache_path` if missing.

    Adds a `sign` column (a copy of `weight`) for consistency with the rest of the
    pipeline (InfluenceGraph autolabeling, `lookup_sign_from_db`), which expect a
    'sign' attribute rather than CollecTRI's native 'weight' column.
    """
    if os.path.exists(cache_path):
        net = pd.read_csv(cache_path)
    else:
        import decoupler as dc

        net = dc.op.collectri(organism=organism, remove_complexes=False)
        net.to_csv(cache_path, index=False)
    net["sign"] = net["weight"]
    return net


def in_db(frame: pd.DataFrame, db: pd.DataFrame) -> pd.Series:
    """Boolean mask: is each column of `frame` a regulator (source) or a target in `db`?"""
    is_source = frame.columns.isin(db.source)
    is_target = frame.columns.isin(db.target)
    return pd.Series(fn.reduce(np.logical_or, [is_source, is_target]), index=frame.columns)


def lookup_sign_from_db(regulator: str, target: str, db: pd.DataFrame) -> int:
    """Look up the curated sign for a (regulator, target) edge in `db` (source/target/sign columns).

    Returns +1/-1 if the pair is annotated, or 0 if it isn't — bonesis treats an
    edge with sign=0 as "either monotonicity" and lets the solver pick the role
    from the observed data instead (see bonesis/aeon.py: sign>0 -> 1, sign<0 -> -1,
    else -> both allowed). Useful for data-driven edges (e.g. from FlowGRN) that
    fall outside the curated database.
    """
    match = db[(db.source == regulator) & (db.target == target)]
    if match.empty:
        return 0
    return int(np.sign(match.sign.iloc[0]))
