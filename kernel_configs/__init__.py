"""Auto-discovery registry for kernel benchmark configurations.

Scans this directory for ``*.toml`` files.  For each file the registry:
1. Parses the TOML data (sizes, dtypes, tolerances, edge_sizes, meta flags).
2. Imports the companion ``.py`` module with the same stem name to obtain the
   four required callables: ``input_generator``, ``reference_fn``, ``flops_fn``,
   ``bytes_fn``.
3. Merges everything into a single config dict whose schema matches what
   ``tools/bench.py`` expects in ``KERNEL_CONFIGS``.
"""

from __future__ import annotations

import importlib
import pathlib
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from ._utils import DTYPE_MAP

_PKG_DIR = pathlib.Path(__file__).resolve().parent


def _load_toml(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _parse_sizes(raw: list[dict]) -> list[tuple[str, dict]]:
    return [(entry["label"], entry["params"]) for entry in raw]


def _parse_dtypes(raw: list[str]):
    import torch
    out: list[torch.dtype] = []
    for name in raw:
        if name not in DTYPE_MAP:
            raise ValueError(f"Unknown dtype string '{name}'. Valid: {list(DTYPE_MAP)}")
        out.append(DTYPE_MAP[name])
    return out


def _parse_tolerances(raw: dict):
    import torch
    out: dict[torch.dtype, dict[str, float]] = {}
    for name, tol in raw.items():
        dt = DTYPE_MAP.get(name)
        if dt is None:
            raise ValueError(f"Unknown dtype '{name}' in tolerances section")
        out[dt] = {"atol": float(tol["atol"]), "rtol": float(tol["rtol"])}
    return out


def _build_config(toml_path: pathlib.Path) -> Dict[str, Any]:
    data = _load_toml(toml_path)
    stem = toml_path.stem

    mod = importlib.import_module(f"kernel_configs.{stem}")

    cfg: Dict[str, Any] = {}

    meta = data.get("meta", {})
    if meta.get("multi_output", False):
        cfg["multi_output"] = True

    cfg["test_sizes"] = _parse_sizes(data["test_sizes"])
    cfg["test_dtypes"] = _parse_dtypes(data["test_dtypes"])
    cfg["tolerances"] = _parse_tolerances(data["tolerances"])

    edge_raw = data.get("edge_sizes", [])
    cfg["edge_sizes"] = _parse_sizes(edge_raw) if edge_raw else []

    cfg["input_generator"] = mod.input_generator
    cfg["reference_fn"] = mod.reference_fn
    cfg["flops_fn"] = mod.flops_fn
    cfg["bytes_fn"] = mod.bytes_fn

    return cfg


def _discover() -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    for toml_path in sorted(_PKG_DIR.glob("*.toml")):
        kernel_name = toml_path.stem
        py_path = toml_path.with_suffix(".py")
        if not py_path.exists():
            raise FileNotFoundError(
                f"Kernel config '{toml_path.name}' has no companion "
                f"'{py_path.name}' providing callables"
            )
        configs[kernel_name] = _build_config(toml_path)
    return configs


KERNEL_CONFIGS: Dict[str, Dict[str, Any]] = _discover()
