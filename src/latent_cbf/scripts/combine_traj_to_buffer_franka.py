#!/usr/bin/env python3
"""
Convert FrankaSkittlesEnv trajectory files (scripts/collect_franka_skittles.py) into the robobuf
``buffer.pkl`` that diffusion4robotics/train.py reads, plus ``ac_norm.json`` / ``ob_norm.json``.

Per transition:
  cam_0  front camera  (stored by robobuf as a 256x256 JPEG)
  cam_1  wrist camera
  state  8-d proprio (tcp pos, tcp quat wxyz, gripper width), min/max-normalised to [-1, 1]
  action 7-d env action, min/max-normalised to [-1, 1]

  python scripts/combine_traj_to_buffer_franka.py \
      --input-dir <DATA_ROOT>/trajs/franka_skittles_il --output-dir <DATA_ROOT>/buffers/franka_il
"""

import argparse
import json
import pathlib
import pickle
import sys

import h5py
import numpy as np
from robobuf.buffers import ObsWrapper, ReplayBuffer, Transition
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from configs.paths import BUFFERS_DIR, TRAJS_DIR


def iter_trajs(files):
    """Yields (file name, trajectory key, group); each group is only valid until the next item."""
    for path in files:
        with h5py.File(path, "r") as f:
            for key in sorted(f["trajectories"].keys()):
                yield path.name, key, f["trajectories"][key]


def minmax(x):
    lo, hi = x.min(0), x.max(0)
    span = np.where(hi - lo < 1e-6, 1.0, hi - lo)  # constant dims map to -1 instead of dividing by 0
    return lo, lo + span


def normalise(x, lo, hi):
    return ((x - lo) / (hi - lo) * 2 - 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=pathlib.Path, default=TRAJS_DIR / "franka_skittles_il")
    p.add_argument("--output-dir", type=pathlib.Path, default=BUFFERS_DIR / "franka_il")
    p.add_argument("--front-key", default="observations")
    p.add_argument("--wrist-key", default="wrist_observations")
    args = p.parse_args()

    files = sorted(args.input_dir.glob("*.h5"))
    assert files, f"no .h5 files in {args.input_dir}"
    print(f"{len(files)} files in {args.input_dir}")

    # pass 1: normalisation statistics
    states = np.concatenate([g["states"][:-1] for *_, g in iter_trajs(files)])
    actions = np.concatenate([g["actions"][:] for *_, g in iter_trajs(files)])
    ob_lo, ob_hi = minmax(states)
    ac_lo, ac_hi = minmax(actions)
    print(f"state  min {np.round(ob_lo, 3)}\n       max {np.round(ob_hi, 3)}")
    print(f"action min {np.round(ac_lo, 3)}\n       max {np.round(ac_hi, 3)}")

    # pass 2: build the buffer. obs_t pairs with a_t, so the final observation (T+1) is dropped.
    buf, n_traj, counts = ReplayBuffer(), 0, {}
    n_total = sum(len(h5py.File(x, "r")["trajectories"]) for x in files)
    for fname, key, g in tqdm(iter_trajs(files), total=n_total, desc="trajectories"):
        front, wrist = g[args.front_key][:], g[args.wrist_key][:]
        s = normalise(g["states"][:-1], ob_lo, ob_hi)
        a = normalise(g["actions"][:], ac_lo, ac_hi)
        for t in range(len(a)):
            obs = {"cam_0": front[t], "cam_1": wrist[t], "state": s[t]}
            buf.add(Transition(ObsWrapper(obs), a[t], 0.0), is_first=(t == 0))
        mode = g.attrs.get("mode", "unknown")
        counts[mode] = counts.get(mode, 0) + 1
        n_traj += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "buffer.pkl", "wb") as f:
        pickle.dump(buf.to_traj_list(), f)
    for name, lo, hi in (("ac_norm.json", ac_lo, ac_hi), ("ob_norm.json", ob_lo, ob_hi)):
        with open(args.output_dir / name, "w") as f:
            json.dump({"minimum": lo.tolist(), "maximum": hi.tolist()}, f)
    with open(args.output_dir / "sources.json", "w") as f:
        json.dump({"input_dir": str(args.input_dir), "files": [x.name for x in files], "modes": counts}, f, indent=1)
    print(f"Saved {n_traj} trajectories ({len(buf)} transitions, modes {counts}) to {args.output_dir}/buffer.pkl")


if __name__ == "__main__":
    main()
