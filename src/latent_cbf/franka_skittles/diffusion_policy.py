"""
Run a diffusion policy trained by diffusion4robotics/train.py (via scripts/combine_traj_to_buffer_franka.py data)
in FrankaSkittlesEnv.

The run directory written by train.py holds everything needed: agent_config.yaml, exp_config.yaml,
ac_norm.json, ob_norm.json and the checkpoints. Observations go through the same path as in training:
images are JPEG round-tripped at 256x256 by robobuf, converted to [0, 1] CHW, then resized to 224 and
ImageNet-normalised; the state is min/max-normalised to [-1, 1]. The policy predicts an ``ac_chunk``
of normalised actions; the first ``exec_horizon`` are de-normalised and executed before re-querying.
"""

import glob
import json
import os
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from robobuf.buffers import ObsWrapper

from configs.paths import DIFFUSION4ROBOTICS_DEFAULTS
from data4robotics import transforms
from data4robotics.models.diffusion_unet import DiffusionUnetAgent
from data4robotics.models.resnet import RobomimicResNet

# order of env observation keys for the camera indexes used at conversion time (cam_0, cam_1)
CAM_KEYS = ("image", "wrist_image")


def latest_checkpoint(run_dir) -> str:
    """``*_latest.ckpt`` in run_dir (train.py's rolling checkpoint)."""
    cands = sorted(glob.glob(os.path.join(run_dir, "*_latest.ckpt")))
    assert cands, f"no *_latest.ckpt in {run_dir}"
    return cands[0]


class FrankaDiffusionPolicy:
    def __init__(self, run_dir, checkpoint: Optional[str] = None, device: str = "cuda",
                 exec_horizon: int = 8, eval_diffusion_steps: Optional[int] = None):
        run_dir = Path(run_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        agent_cfg = yaml.safe_load(open(run_dir / "agent_config.yaml"))
        params = yaml.safe_load(open(run_dir / "exp_config.yaml"))["params"]
        defaults = yaml.safe_load(open(DIFFUSION4ROBOTICS_DEFAULTS))
        ac = json.load(open(run_dir / "ac_norm.json"))
        ob = json.load(open(run_dir / "ob_norm.json"))
        self.ac_min, self.ac_max = np.array(ac["minimum"]), np.array(ac["maximum"])
        self.ob_min, self.ob_max = np.array(ob["minimum"]), np.array(ob["maximum"])

        self.cam_indexes = list(params["cam_indexes"])
        self.ac_chunk, self.img_chunk = params["ac_chunk"], params.get("img_chunk", 1)
        self.exec_horizon = min(exec_horizon, self.ac_chunk)
        feat = {k: v for k, v in agent_cfg["features"].items() if k != "class"}
        self.agent = DiffusionUnetAgent(
            features=RobomimicResNet(**feat),
            shared_mlp=[],
            odim=agent_cfg["odim"],
            n_cams=agent_cfg["n_cams"],
            use_obs=agent_cfg.get("use_obs", False),
            ac_dim=agent_cfg["ac_dim"],
            ac_chunk=self.ac_chunk,
            train_diffusion_steps=defaults["train_diffusion_steps"],
            eval_diffusion_steps=eval_diffusion_steps or defaults["eval_diffusion_steps"],
            imgs_per_cam=self.img_chunk,
            dropout=defaults["dropout"],
            share_cam_features=False,
            feat_batch_norm=False,
            noise_net_kwargs=agent_cfg["noise_net_kwargs"],
        )
        self.checkpoint = checkpoint or latest_checkpoint(run_dir)
        state = torch.load(self.checkpoint, map_location=self.device)
        self.agent.load_state_dict(state["model"])
        self.agent.eval().to(self.device)
        self.transform = transforms.get_transform_by_name("preproc", size=feat["img_size"])
        self.reset()

    def reset(self):
        self._queue = []
        self._img_hist = {c: deque(maxlen=self.img_chunk) for c in self.cam_indexes}
        self.n_queries = 0

    def _prep_images(self, obs: Dict) -> Dict[str, torch.Tensor]:
        out = {}
        for i, c in enumerate(self.cam_indexes):
            # same JPEG / 256 px round trip as robobuf applies to the training images
            img = ObsWrapper({"cam_0": obs[CAM_KEYS[c]]}).image(0)
            hist = self._img_hist[c]
            hist.appendleft(img)  # training stacks frames newest first: [t, t-1, ...]
            while len(hist) < self.img_chunk:
                hist.append(img)
            x = torch.from_numpy(np.stack(hist).copy()).permute(0, 3, 1, 2).float() / 255
            out[f"cam{i}"] = self.transform(x)[None].to(self.device)  # (1, T, 3, H, W)
        return out

    def _prep_state(self, obs: Dict) -> torch.Tensor:
        s = (np.asarray(obs["state"], float) - self.ob_min) / (self.ob_max - self.ob_min) * 2 - 1
        return torch.from_numpy(s.astype(np.float32))[None].to(self.device)

    @torch.no_grad()
    def predict_chunk(self, obs: Dict) -> np.ndarray:
        """Full predicted chunk (ac_chunk, ac_dim) in env units, without touching the execution queue."""
        a = self.agent.get_actions(self._prep_images(obs), self._prep_state(obs))[0].cpu().numpy()
        return (a + 1) / 2 * (self.ac_max - self.ac_min) + self.ac_min

    def act(self, obs: Dict) -> np.ndarray:
        if not self._queue:
            self._queue = list(self.predict_chunk(obs)[:self.exec_horizon])
            self.n_queries += 1
        elif self.img_chunk > 1:
            self._prep_images(obs)  # keep the frame history current between queries
        a = np.clip(self._queue.pop(0), -1, 1)
        a[6] = 1.0 if a[6] > 0 else -1.0  # gripper is binary in the demos
        return a.astype(np.float32)
