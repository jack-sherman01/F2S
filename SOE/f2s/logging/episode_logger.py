import glob
import os
import re
from typing import Any, Dict, Optional

import numpy as np

from f2s.common.io import save_json


class EpisodeLogger:
    """Method-independent episode logger.

    Every method (SOE baseline, Success-only, Failure Replay, F2S, ...)
    must go through this class so downstream failure extraction, world-model
    training, and metrics code never need to know which method produced an
    episode. One episode is saved as a pair of files:

        episodes/episode_{index:06d}.npz   (arrays: actions, states, rewards,
                                             dones, and one obs_<key> array
                                             per observation key)
        episodes/episode_{index:06d}.json  (metadata, see
                                             f2s.common.schemas.EPISODE_META_FIELDS)
    """

    def __init__(self, output_dir: str, task: str, seed: int, round_id: int):
        self.output_dir = output_dir
        self.task = task
        self.seed = seed
        self.round_id = round_id
        self.episodes_dir = os.path.join(output_dir, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)
        self._next_index = self._infer_next_index()
        self._cur: Optional[Dict[str, Any]] = None

    def _infer_next_index(self) -> int:
        existing = glob.glob(os.path.join(self.episodes_dir, "episode_*.json"))
        if not existing:
            return 0
        indices = []
        for path in existing:
            m = re.search(r"episode_(\d+)\.json$", os.path.basename(path))
            if m:
                indices.append(int(m.group(1)))
        return (max(indices) + 1) if indices else 0

    def start_episode(self, episode_id: Optional[str] = None) -> str:
        idx = self._next_index
        self._next_index += 1
        eid = episode_id if episode_id is not None else f"episode_{idx:06d}"
        self._cur = dict(
            episode_id=eid,
            index=idx,
            observations=[],
            states=[],
            actions=[],
            rewards=[],
            dones=[],
            infos=[],
        )
        return eid

    def add_step(self, observation, state, action, reward, done, info=None) -> None:
        if self._cur is None:
            raise RuntimeError("start_episode() must be called before add_step()")
        self._cur["observations"].append(observation)
        self._cur["states"].append(np.asarray(state))
        self._cur["actions"].append(np.asarray(action))
        self._cur["rewards"].append(reward)
        self._cur["dones"].append(bool(done))
        self._cur["infos"].append(info if info is not None else {})

    def finish_episode(
        self,
        success: bool,
        failure_type: str = "success",
        failure_time: Optional[int] = None,
        failure_stage: str = "none",
    ):
        if self._cur is None:
            raise RuntimeError("start_episode() must be called before finish_episode()")
        self._cur["success"] = bool(success)
        self._cur["failure_type"] = failure_type
        self._cur["failure_time"] = failure_time
        self._cur["failure_stage"] = failure_stage
        return self.save()

    def save(self):
        cur = self._cur
        if cur is None:
            raise RuntimeError("nothing to save: call start_episode()/finish_episode() first")

        eid = cur["episode_id"]
        episode_length = len(cur["actions"])

        obs_keys = None
        if episode_length > 0 and isinstance(cur["observations"][0], dict):
            obs_keys = list(cur["observations"][0].keys())

        arrays = dict(
            actions=np.stack(cur["actions"]) if episode_length > 0 else np.zeros((0,)),
            states=np.stack(cur["states"]) if episode_length > 0 else np.zeros((0,)),
            rewards=np.asarray(cur["rewards"], dtype=np.float32),
            dones=np.asarray(cur["dones"], dtype=bool),
        )
        if obs_keys is not None:
            for k in obs_keys:
                arrays[f"obs_{k}"] = np.stack([o[k] for o in cur["observations"]])
        elif episode_length > 0:
            arrays["observations"] = np.stack(cur["observations"])

        npz_path = os.path.join(self.episodes_dir, f"{eid}.npz")
        np.savez_compressed(npz_path, **arrays)

        meta = dict(
            episode_id=eid,
            task=self.task,
            seed=self.seed,
            round=self.round_id,
            success=bool(cur["success"]),
            failure_type=cur["failure_type"],
            failure_time=cur["failure_time"],
            failure_stage=cur["failure_stage"],
            episode_length=episode_length,
            obs_keys=obs_keys,
        )
        json_path = os.path.join(self.episodes_dir, f"{eid}.json")
        save_json(json_path, meta)

        self._cur = None
        return npz_path, json_path


def load_episode(episodes_dir: str, episode_id: str):
    """Load one episode back into (metadata dict, arrays dict)."""
    from f2s.common.io import load_json

    meta = load_json(os.path.join(episodes_dir, f"{episode_id}.json"))
    with np.load(os.path.join(episodes_dir, f"{episode_id}.npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return meta, arrays
