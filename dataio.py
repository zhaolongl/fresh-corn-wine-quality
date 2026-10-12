"""Read the reviewed publication data from JSON, without executable pickles.

The resources are processed tables and saved analysis results, not raw instrument
records. See ``resources/provenance.json`` for their origin and limitations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
_BUNDLES = frozenset({"paper", "aroma", "ml", "aroma_candidates", "extra"})


def resource_path(relative: str) -> Path:
    """Return a package resource path; reject paths outside the resource folder."""
    path = (_RESOURCE_ROOT / relative).resolve()
    if not path.is_relative_to(_RESOURCE_ROOT.resolve()):
        raise ValueError("Resource path must remain inside the resource folder")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _decode(value):
    """Decode only the explicitly documented JSON data types."""
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind is None:
        return {key: _decode(item) for key, item in value.items()}
    if kind == "float":
        return {"nan": float("nan"), "inf": float("inf"), "-inf": -float("inf")}[value["value"]]
    if kind == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if kind == "mapping":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    if kind == "ndarray":
        return np.asarray(_decode(value["values"]), dtype=value["dtype"]).reshape(value["shape"])
    if kind == "range_index":
        return pd.RangeIndex(value["start"], value["stop"], value["step"], name=_decode(value["name"]))
    if kind == "index":
        return pd.Index(_decode(value["values"]), dtype=value["dtype"], name=_decode(value["name"]), tupleize_cols=False)
    if kind == "dataframe":
        frame = pd.DataFrame(_decode(value["values"]), index=_decode(value["index"]), columns=_decode(value["columns"]))
        for position, dtype in enumerate(value["dtypes"]):
            frame.isetitem(position, frame.iloc[:, position].astype(dtype))
        return frame
    if kind == "series":
        return pd.Series(_decode(value["values"]), index=_decode(value["index"]), dtype=value["dtype"], name=_decode(value["name"]))
    raise ValueError(f"Unsupported JSON data type: {kind!r}")


def load_bundle(name: str) -> dict:
    """Load a fresh mutable bundle: paper, aroma, ml, aroma_candidates or extra.

    Every call returns independent objects so plotting cannot mutate a shared
    global cache. No data correction or model execution occurs while loading.
    """
    if name not in _BUNDLES:
        raise ValueError(f"Unknown bundle {name!r}; choose from {sorted(_BUNDLES)}")
    with resource_path(f"bundles/{name}.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or payload.get("bundle") != name:
        raise ValueError(f"Invalid or unsupported bundle document: {name}")
    return _decode(payload["data"])
