"""
Waypoint-following expert for FrankaSkittlesEnv.

mode="safe":   grasp near the open end -> lifting tips the closed end down, candies stay in.
mode="unsafe": grasp near the closed end -> lifting swings the open end down and spills.
mode="random": grasp anywhere along the bag; the physics decides whether it spills.

With ``randomize=True`` every ``reset()`` samples a new grasp point, approach,
lift/carry path, speed, pauses and grasp yaw, so demos of the same mode differ.
Gaussian ``noise`` is added to every action on top of that.
"""

from typing import List, Tuple

import mujoco
import numpy as np


class ScriptedExpert:
    def __init__(self, env, mode: str = "safe", randomize: bool = True, noise: float = 0.05, rng=None):
        assert mode in ("safe", "unsafe", "random")
        self.env, self.mode = env, mode
        self.randomize = randomize
        self.noise = noise
        self.rng = rng or np.random.default_rng()

    def _sample(self, low, high, default):
        return self.rng.uniform(low, high) if self.randomize else default

    def reset(self) -> None:
        env = self.env
        l = env.bag_size[1]
        inset = self._sample(0.02, 0.06, 0.03)  # grasp distance from the chosen end
        if self.mode == "safe":
            end_y = -(l / 2 - inset)
        elif self.mode == "unsafe":
            end_y = l / 2 - inset
        else:
            end_y = self.rng.uniform(-(l / 2 - 0.02), l / 2 - 0.02)
        self.grasp_y = end_y  # bag frame; open end is at -l/2

        grasp = env.bag_point(np.array([0.0, end_y, 0.0]))
        grasp[2] = self._sample(0.025, 0.035, 0.03)
        above = grasp + [0, 0, self._sample(0.08, 0.15, 0.12)]
        # approach from a slightly offset point so descents aren't perfectly vertical
        above[:2] += self.rng.uniform(-0.02, 0.02, 2) if self.randomize else 0.0

        lift = np.array([self._sample(-0.15, 0.05, -0.10), self._sample(-0.08, 0.08, 0.0),
                         self._sample(0.15, 0.30, 0.25)])
        carry = lift + (np.array([self.rng.uniform(-0.10, 0.10), self.rng.uniform(-0.15, 0.15),
                                  self.rng.uniform(-0.05, 0.05)]) if self.randomize else 0.0)

        # (target position, gripper closed?, steps to hold after reaching, speed scale)
        self.plan: List[Tuple[np.ndarray, bool, int, float]] = [
            (above, False, int(self._sample(0, 4, 0)), self._sample(0.6, 1.0, 1.0)),
            (grasp, False, int(self._sample(1, 6, 3)), self._sample(0.4, 0.8, 1.0)),
            (grasp, True, int(self._sample(5, 12, 8)), 1.0),
            (grasp + lift, True, int(self._sample(0, 6, 0)), self._sample(0.4, 1.0, 1.0)),
            (grasp + carry, True, int(self._sample(10, 20, 15)), self._sample(0.4, 1.0, 1.0)),
        ]
        yaw_offset = self._sample(-0.15, 0.15, 0.0)
        self.quat = self._grasp_quat(env.bag_yaw() + yaw_offset, env.target_quat)
        self.idx, self.hold = 0, 0

    @staticmethod
    def _grasp_quat(bag_yaw: float, current_quat: np.ndarray) -> np.ndarray:
        """Hand z down, finger axis (hand y) across the bag width so the fingers pinch the side walls."""
        # the gripper is symmetric: of the two finger directions, take the one closer to the current hand
        best, best_dot = None, -1.0
        for yaw in (bag_yaw, bag_yaw + np.pi):
            y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            z = np.array([0.0, 0.0, -1.0])
            rot = np.stack([np.cross(y, z), y, z], axis=1)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, rot.flatten())
            dot = abs(np.dot(q, current_quat))
            if dot > best_dot:
                best, best_dot = q, dot
        return best

    @property
    def done(self) -> bool:
        return self.idx >= len(self.plan)

    def act(self) -> np.ndarray:
        env = self.env
        target, close, hold_steps, speed = self.plan[min(self.idx, len(self.plan) - 1)]
        dpos = target - env.target_pos
        rotvec = np.zeros(3)
        mujoco.mju_subQuat(rotvec, self.quat, env.target_quat)
        cur = np.zeros(9)
        mujoco.mju_quat2Mat(cur, env.target_quat)
        rotvec = cur.reshape(3, 3) @ rotvec

        action = np.zeros(7)
        action[:3] = np.clip(dpos / env.max_pos_step, -speed, speed)
        action[3:6] = np.clip(rotvec / env.max_rot_step, -1, 1)
        action[6] = 1.0 if close else -1.0
        if self.noise > 0:
            action[:6] += self.rng.normal(0, self.noise, 6)

        # arrival tolerance grows with the noise, otherwise noisy steps keep jumping past it
        pos_tol = max(0.004, 2 * self.noise * env.max_pos_step)
        rot_tol = max(0.03, 2 * self.noise * env.max_rot_step)
        if not self.done and np.linalg.norm(dpos) < pos_tol and np.linalg.norm(rotvec) < rot_tol:
            self.hold += 1
            if self.hold > hold_steps:
                self.idx, self.hold = self.idx + 1, 0
        return np.clip(action, -1, 1)
