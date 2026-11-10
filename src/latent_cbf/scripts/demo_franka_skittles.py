#!/usr/bin/env python3
"""
Roll out the scripted expert in FrankaSkittlesEnv.

Save a gif per mode (front | wrist camera side by side) to visualizations/:
  python scripts/demo_franka_skittles.py --seed 0

Spill statistics over many randomized episodes (no gifs):
  python scripts/demo_franka_skittles.py --stats 50
"""

import argparse
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np

from franka_skittles.franka_skittles_env import FrankaSkittlesEnv
from franka_skittles.scripted_expert import ScriptedExpert

MODES = ("safe", "unsafe", "random")


def rollout(env, expert, seed: int, extra_steps: int = 20, record: bool = False):
    obs, info = env.reset(seed=seed)
    expert.reset()
    frames, n_spilled = [], []
    for _ in range(env.max_episode_steps):
        obs, _, _, truncated, info = env.step(expert.act())
        if record:
            frames.append(np.concatenate([obs["image"], obs["wrist_image"]], axis=1))
        n_spilled.append(info["n_spilled"])
        if expert.done:
            extra_steps -= 1
        if truncated or extra_steps <= 0:
            break
    first_fail = next((t for t, n in enumerate(n_spilled) if n > 0), None)
    return frames, n_spilled[-1], first_fail, len(n_spilled), expert.done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--no-randomize", action="store_true", help="Nominal expert parameters (bag pose still random)")
    p.add_argument("--stats", type=int, default=0, help="Run N episodes per mode and print spill statistics")
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("visualizations"))
    args = p.parse_args()

    env = FrankaSkittlesEnv(image_size=64 if args.stats else args.image_size, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    if args.stats:
        for mode in MODES:
            expert = ScriptedExpert(env, mode=mode, randomize=not args.no_randomize, noise=args.noise, rng=rng)
            spilled, lengths, unfinished = [], [], 0
            for ep in range(args.stats):
                _, n, _, length, done = rollout(env, expert, seed=args.seed * 10_000 + ep)
                spilled.append(n)
                lengths.append(length)
                unfinished += not done
            spilled = np.array(spilled)
            print(f"{mode:6s}: spill rate {np.mean(spilled > 0):.0%} ({np.sum(spilled > 0)}/{args.stats}), "
                  f"mean candies spilled {spilled.mean():.1f}/{env.n_candies}, "
                  f"episode length {np.mean(lengths):.0f}±{np.std(lengths):.0f}, expert unfinished {unfinished}")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for mode in MODES:
            expert = ScriptedExpert(env, mode=mode, randomize=not args.no_randomize, noise=args.noise, rng=rng)
            frames, n, first_fail, length, done = rollout(env, expert, args.seed, record=True)
            path = args.out_dir / f"franka_skittles_{mode}_seed{args.seed}.gif"
            imageio.mimsave(path, frames, fps=env.metadata["render_fps"])
            print(f"{mode:6s}: {length} steps, spilled {n}/{env.n_candies}, failure from step {first_fail}, "
                  f"expert finished={done} -> {path}")
    env.close()


if __name__ == "__main__":
    main()
