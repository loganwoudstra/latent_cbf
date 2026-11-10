#!/usr/bin/env python3
"""
Evaluate a diffusion IL policy in FrankaSkittlesEnv.

Per episode it records:
  spilled   any candy left the bag (the failure)
  lifted    the bag rose more than --lift-height above the table at some point
  success   lifted, still held up at the end, and nothing spilled
  stalled   never lifted
  grasp_y   tcp position along the bag (bag frame, metres) when it first lifts off. The open end is -l/2 = -0.08,
            the closed end +0.08. Grasps with grasp_y < 0 are on the open-end half.

  python scripts/eval_franka_il.py --run-dir <DATA_ROOT>/diffusion/franka_il --episodes 50 --gifs 5
  # also keep the rollouts (same h5 layout as collect_franka_skittles.py), e.g. as world-model data:
  python scripts/eval_franka_il.py --run-dir ... --episodes 200 --save-dir <DATA_ROOT>/trajs/franka_skittles_policy
"""

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio
import numpy as np

from franka_skittles.diffusion_policy import FrankaDiffusionPolicy
from franka_skittles.franka_skittles_env import FrankaSkittlesEnv
from franka_skittles.trajectory_io import EpisodeRecorder, write_episode, write_footer, write_header


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=pathlib.Path, required=True, help="train.py checkpoint dir")
    p.add_argument("--checkpoint", type=str, default=None, help="Default: <run-dir>/*_latest.ckpt")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=1000, help="Episode i uses env seed seed*1_000_000 + i "
                   "(demos used seeds 1 and 2, so the default doesn't repeat training bag poses)")
    p.add_argument("--max-steps", type=int, default=150, help="Demos are ~100 steps")
    p.add_argument("--exec-horizon", type=int, default=8, help="Actions executed per policy query")
    p.add_argument("--diffusion-steps", type=int, default=None, help="Default: defaults.yaml eval_diffusion_steps")
    p.add_argument("--image-size", type=int, default=128, help="Must match the training data")
    p.add_argument("--lift-height", type=float, default=0.05)
    p.add_argument("--gifs", type=int, default=0, help="Save gifs of the first N episodes")
    p.add_argument("--gif-dir", type=pathlib.Path, default=pathlib.Path("visualizations/franka_il_eval"))
    p.add_argument("--save-dir", type=pathlib.Path, default=None, help="Also write rollouts as trajectory h5 files")
    p.add_argument("--episodes-per-file", type=int, default=50)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run_episode(env, policy, seed, args, record_frames=False):
    obs, info = env.reset(seed=seed)
    policy.reset()
    rec = EpisodeRecorder(obs, info)
    table_z = info["bag_pos"][2]
    frames, grasp_y, lifted = [], None, False
    for t in range(args.max_steps):
        a = policy.act(obs)
        obs, _, _, truncated, info = env.step(a)
        rec.add(a, obs, info)
        if record_frames:
            frames.append(np.concatenate([obs["image"], obs["wrist_image"]], 1))
        if not lifted and info["bag_pos"][2] - table_z > args.lift_height:
            lifted = True
            R = env.data.xmat[env.bag_body].reshape(3, 3)
            grasp_y = float(((env.data.site_xpos[env.tcp_site] - env.data.xpos[env.bag_body]) @ R)[1])
        if truncated:
            break
    held = bool(info["bag_pos"][2] - table_z > args.lift_height)
    spilled = bool(info["failure"])
    ep = rec.finish(episode_idx=int(seed % 1_000_000), seed=int(seed), mode="policy", noise=0.0,
                    success=bool(lifted and held and not spilled), lifted=lifted, held_at_end=held,
                    grasp_y=np.nan if grasp_y is None else grasp_y, n_policy_queries=policy.n_queries)
    return ep, frames


def main():
    args = parse_args()
    policy = FrankaDiffusionPolicy(args.run_dir, args.checkpoint, device=args.device,
                                   exec_horizon=args.exec_horizon, eval_diffusion_steps=args.diffusion_steps)
    print(f"Policy: {policy.checkpoint} (chunk {policy.ac_chunk}, executing {policy.exec_horizon})")
    env = FrankaSkittlesEnv(image_size=args.image_size, max_episode_steps=args.max_steps)
    if args.gifs:
        args.gif_dir.mkdir(parents=True, exist_ok=True)

    h5, trajs, file_idx, file_eps, results = None, None, 0, [], []
    t0 = time.time()
    for i in range(args.episodes):
        seed = args.seed * 1_000_000 + i
        ep, frames = run_episode(env, policy, seed, args, record_frames=i < args.gifs)
        a = ep["attrs"]
        results.append(a)
        side = "-" if np.isnan(a["grasp_y"]) else ("open" if a["grasp_y"] < 0 else "closed")
        print(f"[{i + 1}/{args.episodes}] seed {seed}: {'SPILL' if a['collision'] else 'ok   '} "
              f"lifted={a['lifted']} success={a['success']} grasp_y={a['grasp_y']:+.3f} ({side} half) "
              f"spilled {a['n_spilled']}/{env.n_candies}  [{time.time() - t0:.0f}s]", flush=True)
        if frames:
            imageio.mimsave(args.gif_dir / f"ep{i:03d}_seed{seed}.gif", frames, fps=env.metadata["render_fps"])

        if args.save_dir is not None:
            if h5 is None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                h5 = h5py.File(args.save_dir / f"franka_policy_{file_idx:04d}.h5.tmp", "w")
                trajs = write_header(h5, {k: v for k, v in vars(args).items()}, env,
                                     dict(type="diffusion_policy", checkpoint=str(policy.checkpoint),
                                          exec_horizon=policy.exec_horizon))
            write_episode(trajs, len(file_eps), ep)
            file_eps.append(a)
            if len(file_eps) == args.episodes_per_file or i == args.episodes - 1:
                write_footer(h5, file_eps)
                name = h5.filename
                h5.close()
                os.rename(name, name[:-len(".tmp")])
                h5, file_eps, file_idx = None, [], file_idx + 1

    r = results
    gy = np.array([x["grasp_y"] for x in r], float)
    lifted = np.array([x["lifted"] for x in r])
    spill = np.array([x["collision"] for x in r])
    summary = dict(
        episodes=len(r), checkpoint=str(policy.checkpoint),
        spill_rate=float(spill.mean()), success_rate=float(np.mean([x["success"] for x in r])),
        stall_rate=float(1 - lifted.mean()),
        open_half_grasps=int(np.sum(gy[lifted] < 0)), closed_half_grasps=int(np.sum(gy[lifted] >= 0)),
        spill_rate_open_half=float(spill[lifted & (gy < 0)].mean()) if np.any(lifted & (gy < 0)) else None,
        spill_rate_closed_half=float(spill[lifted & (gy >= 0)].mean()) if np.any(lifted & (gy >= 0)) else None,
    )
    print("\n" + json.dumps(summary, indent=1))
    with open(args.run_dir / f"eval_{pathlib.Path(policy.checkpoint).stem}_seed{args.seed}.json", "w") as f:
        json.dump(dict(summary=summary, episodes=[{k: (v.item() if hasattr(v, "item") else v) for k, v in x.items()}
                                                  for x in r]), f, indent=1)
    env.close()


if __name__ == "__main__":
    main()
