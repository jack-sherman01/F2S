import glob
import json
import os
from typing import Any, Dict, List

import numpy as np


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def list_episode_metadata(episodes_dir: str) -> List[str]:
    """Return sorted paths to every episode_*.json metadata file in a directory."""
    return sorted(glob.glob(os.path.join(episodes_dir, "*.json")))


def load_all_episode_metadata(episodes_dir: str) -> List[Dict[str, Any]]:
    return [load_json(p) for p in list_episode_metadata(episodes_dir)]


def git_commit_hash(repo_dir: str) -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def ensure_fresh_dir(path: str) -> None:
    """Refuse to silently reuse an existing experiment directory (proposal rule:
    'Never overwrite an experiment directory')."""
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(
            f"Refusing to write into non-empty experiment directory: {path}. "
            "Use a new round/repetition/seed directory instead."
        )
    os.makedirs(path, exist_ok=True)
