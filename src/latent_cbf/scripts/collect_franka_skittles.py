#!/usr/bin/env python3
"""
Collect FrankaSkittlesEnv trajectories with the scripted expert and save them as HDF5.

Uses the same layout as scripts/collect_trajs.py (Dubins), so the existing tooling
(combine_trajectories.py, h5 viewers) can read the files:

    metadata/                      attrs: config, collection_timestamp, n_trajectories, complete
      environment/                 attrs: env constructor kwargs
      controller/                  attrs: type, mode_probs, noise_levels, perturb_*
    statistics/                    attrs: success_rate, collision_rate, average_steps, average_reward
    trajectories/trajectory_XXXX/
      observations        (T+1, H, W, 3) uint8    front camera
      wrist_observations  (T+1, H, W, 3) uint8    wrist camera
      states              (T+1, 8)  float32       proprio: tcp pos (3), tcp quat wxyz (4), gripper width (1)
      initial_state       (8,)      float32
      privileged_states   (T+1, 8)  float32       bag pos (3), bag quat (4), n_spilled (1)
      actions             (T, 7)    float32       executed action a_t in [-1, 1]
      rewards             (T,)      float64       zeros (no task reward)
      failures            (T,)      bool          failure *after* a_t (sticky once a candy spills)
      perturbed           (T,)      bool          a_t came from a random-action burst
      attrs: episode_idx, seed, mode, noise, success, collision, steps, total_reward,
             final_distance_to_goal, n_spilled, first_failure_step, expert_finished, n_perturbed

"collision" means "failure" (spill) so Dubins tools that read it still work.

Each worker writes its own shard files to <DATA_ROOT>/trajs/franka_skittles/ by default. That is a subdirectory,
so combine_trajectories.py (which globs trajs/*.h5) doesn't mix it with the Dubins data.
A shard is written to *.h5.tmp and renamed when it is complete; rerunning the same command skips finished shards.

Examples:
  # quick format check (2 episodes, scratch dir)
  python scripts/collect_franka_skittles.py --n-episodes 2 --episodes-per-shard 2 --workers 1 --out-dir /tmp/fs_test
  # full collection
  python scripts/collect_franka_skittles.py --n-episodes 1000 --workers 8
"""

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # MuJoCo/numpy thread oversubscription makes workers slower

import numpy as np

from configs.paths import TRAJS_DIR
from franka_skittles.trajectory_io import EpisodeRecorder, write_episode, write_footer, write_header

MODES = ("safe", "unsafe", "random")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-episodes", type=int, default=1000)
    p.add_argument("--episodes-per-shard", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="Base seed; episode i uses seed*1_000_000 + i")
    p.add_argument("--out-dir", type=pathlib.Path, default=TRAJS_DIR / "franka_skittles")
    p.add_argument("--prefix", type=str, default="franka_skittles")
    p.add_argument("--image-size", type=int, default=128)
    # data mix
    p.add_argument("--mode-probs", type=float, nargs=3, default=(0.4, 0.3, 0.3),
                   metavar=("SAFE", "UNSAFE", "RANDOM"), help="Probability of each expert mode per episode")
    p.add_argument("--noise-levels", type=float, nargs="+", default=(0.0, 0.05, 0.1, 0.2),
                   help="Per-episode Gaussian action noise is drawn uniformly from these")
    p.add_argument("--perturb-prob", type=float, default=0.3,
                   help="Chance an episode gets random-action bursts (off-policy data for the WM)")
    p.add_argument("--perturb-bursts", type=int, nargs=2, default=(1, 3), metavar=("MIN", "MAX"))
    p.add_argument("--perturb-len", type=int, nargs=2, default=(3, 10), metavar=("MIN", "MAX"))
    p.add_argument("--perturb-gripper-prob", type=float, default=0.1,
                   help="Within a burst, chance the gripper command is also randomised")
    # episode length
    p.add_argument("--extra-steps", type=int, default=20, help="Steps recorded after the expert finishes")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--compression", type=int, default=4, help="gzip level (0 = off)")
    args = p.parse_args(argv)
    probs = np.asarray(args.mode_probs, float)
    assert np.all(probs >= 0) and probs.sum() > 0
    args.mode_probs = tuple(probs / probs.sum())
    return args


def env_kwargs(args):
    return dict(image_size=args.image_size, max_episode_steps=args.max_steps)


def run_episode(env, expert_cls, args, episode_idx):
    """Roll out one episode. Returns a dict of arrays and attrs."""
    seed = args.seed * 1_000_000 + episode_idx
    rng = np.random.default_rng(seed)
    mode = MODES[rng.choice(3, p=args.mode_probs)]
    noise = float(rng.choice(args.noise_levels))
    expert = expert_cls(env, mode=mode, randomize=True, noise=noise, rng=rng)

    # random-action bursts: start steps sampled within the expected episode length
    bursts = []
    if rng.random() < args.perturb_prob:
        n = rng.integers(args.perturb_bursts[0], args.perturb_bursts[1] + 1)
        starts = np.sort(rng.integers(5, 90, size=n))
        bursts = [(int(s), int(rng.integers(args.perturb_len[0], args.perturb_len[1] + 1))) for s in starts]
    burst_steps = {t for s, L in bursts for t in range(s, s + L)}
    burst_gripper = rng.random() < args.perturb_gripper_prob

    obs, info = env.reset(seed=seed)
    expert.reset()
    rec = EpisodeRecorder(obs, info)
    extra = args.extra_steps
    for t in range(env.max_episode_steps):
        a = expert.act()  # always called so the expert keeps tracking its plan
        if t in burst_steps:
            a = a.copy()
            a[:6] = rng.uniform(-1, 1, 6)
            if burst_gripper:
                a[6] = rng.choice([-1.0, 1.0])
        obs, _, _, truncated, info = env.step(a)
        rec.add(a, obs, info, perturbed=t in burst_steps)
        if expert.done:
            extra -= 1
        if truncated or extra <= 0:
            break

    return rec.finish(episode_idx=episode_idx, seed=seed, mode=mode, noise=noise,
                      success=bool(expert.done and not info["failure"]), expert_finished=bool(expert.done),
                      bursts=json.dumps(bursts), burst_gripper=bool(burst_gripper))


def collect_shard(job):
    """Worker entry point: collect episodes [start, stop) into one shard file."""
    args, shard_idx, start, stop = job
    import h5py
    from franka_skittles.franka_skittles_env import FrankaSkittlesEnv
    from franka_skittles.scripted_expert import ScriptedExpert

    path = args.out_dir / f"{args.prefix}_{shard_idx:04d}.h5"
    if path.exists():
        return str(path), None  # finished in an earlier run
    tmp = path.with_suffix(".h5.tmp")

    env = FrankaSkittlesEnv(**env_kwargs(args))
    t0 = time.time()
    summary = []
    with h5py.File(tmp, "w") as f:
        trajs = write_header(f, vars(args), env, dict(type="scripted_expert", mode_probs=args.mode_probs,
                                                      noise_levels=args.noise_levels, perturb_prob=args.perturb_prob))
        for i, ep in enumerate(range(start, stop)):
            ep_out = run_episode(env, ScriptedExpert, args, ep)
            write_episode(trajs, i, ep_out, args.compression)
            summary.append(ep_out["attrs"])
        write_footer(f, summary)
    env.close()
    tmp.rename(path)
    return str(path), dict(summary=summary, seconds=time.time() - t0)


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = int(np.ceil(args.n_episodes / args.episodes_per_shard))
    jobs = [(args, s, s * args.episodes_per_shard, min((s + 1) * args.episodes_per_shard, args.n_episodes))
            for s in range(n_shards)]
    print(f"Collecting {args.n_episodes} episodes in {n_shards} shards with {args.workers} workers -> {args.out_dir}")

    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(args.workers)  # spawn: fresh EGL context per worker
        results = pool.imap_unordered(collect_shard, jobs)
    else:
        pool, results = None, map(collect_shard, jobs)

    all_eps = []
    for done, (path, res) in enumerate(results, 1):
        if res is None:
            print(f"[{done}/{n_shards}] skip (exists) {path}")
            continue
        s = res["summary"]
        all_eps += s
        print(f"[{done}/{n_shards}] {path}: {len(s)} eps in {res['seconds']:.0f}s, "
              f"spill {np.mean([e['collision'] for e in s]):.0%}")
    if pool is not None:
        pool.close(); pool.join()

    if all_eps:
        print("\nSummary (episodes collected this run):")
        for mode in MODES:
            m = [e for e in all_eps if e["mode"] == mode]
            if m:
                print(f"  {mode:6s}: {len(m):5d} eps, spill rate {np.mean([e['collision'] for e in m]):.0%}, "
                      f"mean length {np.mean([e['steps'] for e in m]):.0f}")
        print(f"  perturbed episodes: {np.mean([e['n_perturbed'] > 0 for e in all_eps]):.0%}, "
              f"overall spill rate {np.mean([e['collision'] for e in all_eps]):.0%}")


if __name__ == "__main__":
    main()
