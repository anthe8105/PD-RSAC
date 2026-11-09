"""Run provenance: resolved config, CLI arguments and a manifest (audit D04, D40, F46, N6).

The same record is written next to the training logs and embedded in every checkpoint,
so evaluation can rebuild exactly the configuration a model was trained with.
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from ..config import Config, ConfigLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES = ("torch", "numpy", "pandas", "pyarrow", "h3", "scipy", "gurobipy", "PyYAML")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _git_state() -> Dict[str, Any]:
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def _versions() -> Dict[str, Optional[str]]:
    out = {"python": platform.python_version()}
    for pkg in _PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def file_digest(path: str, chunk: int = 8 << 20) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"path": str(path), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return {"path": str(p.resolve()), "exists": True, "bytes": p.stat().st_size, "sha256": h.hexdigest()}


def spatial_digest(hex_ids: Optional[Iterable[str]]) -> Optional[Dict[str, Any]]:
    if hex_ids is None:
        return None
    ids = [str(h) for h in hex_ids]
    return {"num_hexes": len(ids), "sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def build_run_record(
    config: Config,
    args: Any,
    data_path: Optional[str] = None,
    hex_ids: Optional[Iterable[str]] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect the provenance of a run. Call after the config is final (e.g. num_hexes set from data)."""
    run_args = {k: _jsonable(v) for k, v in sorted(vars(args).items())} if args is not None else {}
    manifest = {
        "created_unix": int(time.time()),
        "argv": list(sys.argv),
        "git": _git_state(),
        "versions": _versions(),
        "seed": run_args.get("seed"),
        "preset": preset,
        "data": file_digest(data_path) if data_path else None,
        "spatial": spatial_digest(hex_ids),
    }
    return {"config": ConfigLoader.to_dict(config), "run_args": run_args, "manifest": manifest}


def write_run_record(run_dir: str, record: Dict[str, Any]) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(record["config"], f, default_flow_style=False, sort_keys=False)
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"run_args": record["run_args"], **record["manifest"]}, f, indent=2)


def checkpoint_fields(trainer: Any) -> Dict[str, Any]:
    """Fields every trainer adds to its checkpoint dict (empty when no record is attached)."""
    record = getattr(trainer, "run_record", None)
    if not record:
        return {}
    return {"config": record["config"], "run_args": record["run_args"], "manifest": record["manifest"]}
