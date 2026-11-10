"""
Record FrankaSkittlesEnv episodes and write them in the trajectory-file layout of
scripts/collect_trajs.py (see scripts/collect_franka_skittles.py for the full schema).
Shared by the scripted-expert collector and the IL-policy evaluator.
"""

import json
import pathlib
from datetime import datetime
from typing import Dict, List

import numpy as np

ENV_ATTRS = ("image_size", "n_candies", "candy_radius", "bag_size", "max_episode_steps", "max_pos_step",
             "max_rot_step", "bag_pos_noise", "bag_yaw_range", "hand_yaw_noise", "hand_tilt_noise",
             "hand_start_pos")


class EpisodeRecorder:
    """Accumulates (obs, action, failure) for one episode; ``finish`` returns {'data', 'attrs'}."""

    def __init__(self, obs: Dict, info: Dict):
        self.images, self.wrist, self.states = [obs["image"]], [obs["wrist_image"]], [obs["state"]]
        self.privileged = [self._priv(info)]
        self.actions, self.failures, self.perturbed = [], [], []
        self.info = info

    @staticmethod
    def _priv(info):
        return np.concatenate([info["bag_pos"], info["bag_quat"], [info["n_spilled"]]]).astype(np.float32)

    def add(self, action, obs: Dict, info: Dict, perturbed: bool = False):
        self.images.append(obs["image"]); self.wrist.append(obs["wrist_image"])
        self.states.append(obs["state"]); self.privileged.append(self._priv(info))
        self.actions.append(np.asarray(action, np.float32)); self.failures.append(bool(info["failure"]))
        self.perturbed.append(bool(perturbed))
        self.info = info

    def finish(self, **attrs) -> Dict:
        failures = np.array(self.failures, bool)
        T = len(self.actions)
        base = dict(collision=bool(failures.any()), steps=T, total_reward=0.0, final_distance_to_goal=0.0,
                    n_spilled=int(self.info["n_spilled"]),
                    first_failure_step=int(np.argmax(failures)) if failures.any() else -1,
                    n_perturbed=int(np.sum(self.perturbed)))
        base.update(attrs)
        return dict(
            data=dict(
                observations=np.stack(self.images).astype(np.uint8),
                wrist_observations=np.stack(self.wrist).astype(np.uint8),
                states=np.stack(self.states).astype(np.float32),
                initial_state=np.asarray(self.states[0], np.float32),
                privileged_states=np.stack(self.privileged),
                actions=np.stack(self.actions),
                rewards=np.zeros(T, np.float64),
                failures=failures,
                perturbed=np.array(self.perturbed, bool),
            ),
            attrs=base,
        )


def write_header(f, config: Dict, env, controller_attrs: Dict):
    """metadata/ (config, environment, controller) and an empty trajectories/ group."""
    meta = f.create_group("metadata")
    meta.attrs["config"] = json.dumps({k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in config.items()})
    meta.attrs["collection_timestamp"] = datetime.now().isoformat(timespec="seconds")
    meta.attrs["complete"] = False
    env_g = meta.create_group("environment")
    env_g.attrs["name"] = "FrankaSkittlesEnv"
    env_g.attrs["control_dt"] = env.n_substeps * env.model.opt.timestep
    for k in ENV_ATTRS:
        env_g.attrs[k] = getattr(env, k)
    ctrl = meta.create_group("controller")
    for k, v in controller_attrs.items():
        ctrl.attrs[k] = v
    return f.create_group("trajectories")


def write_episode(trajs, i: int, ep: Dict, compression: int = 4):
    comp = dict(compression="gzip", compression_opts=compression) if compression > 0 else {}
    g = trajs.create_group(f"trajectory_{i:04d}")
    for k, v in ep["attrs"].items():
        g.attrs[k] = v
    for k, v in ep["data"].items():
        if v.ndim == 0 or k == "initial_state":
            g.create_dataset(k, data=v)
        elif v.ndim == 4:  # images: chunk per frame
            g.create_dataset(k, data=v, chunks=(1, *v.shape[1:]), **comp)
        else:
            g.create_dataset(k, data=v, **comp)


def write_footer(f, summary: List[Dict]):
    stats = f.create_group("statistics")
    stats.attrs["success_rate"] = float(np.mean([s["success"] for s in summary]))
    stats.attrs["collision_rate"] = float(np.mean([s["collision"] for s in summary]))
    stats.attrs["average_steps"] = float(np.mean([s["steps"] for s in summary]))
    stats.attrs["average_reward"] = 0.0
    f["metadata"].attrs["n_trajectories"] = len(summary)
    f["metadata"].attrs["complete"] = True
