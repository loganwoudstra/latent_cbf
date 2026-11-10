"""
FrankaSkittlesEnv - Franka Panda picking up an open "bag" of candies in MuJoCo.

Sim analogue of the hardware task in Nakamura et al. (2025): the bag is a rigid
box lying on its side with one open end, filled with small spheres. Grasping the
closed end and lifting swings the open end down and spills the candies; grasping
the open end keeps them inside. ``info['failure']`` is set (and stays set) once
any candy leaves the bag.

Action: 7-D [dx, dy, dz, drx, dry, drz, gripper] -- end-effector delta in the
world frame (metres / axis-angle radians, clipped per step) plus gripper
(> 0 closes). Converted to joint targets with damped least-squares IK.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from configs.paths import PANDA_XML

HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034  # flange -> fingertip centre along the hand z-axis


def _look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera xyaxes for a camera at ``pos`` looking at ``target``."""
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return [*right, *np.cross(right, forward)]


class FrankaSkittlesEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self,
                 image_size: int = 128,
                 n_candies: int = 30,
                 candy_radius: float = 0.007,
                 bag_size: Tuple[float, float, float] = (0.06, 0.16, 0.05),  # width (x), length (y), height (z)
                 wall: float = 0.005,
                 control_dt: float = 0.05,
                 max_episode_steps: int = 200,
                 max_pos_step: float = 0.02,
                 max_rot_step: float = 0.1,
                 bag_pos_noise: float = 0.05,
                 # yaw ~ U(range). Yaw 0 puts the open end at -y. Across [-15, 105] deg the open end faces the
                 # front camera (>= 25 px of the liner at 128 px anywhere within bag_pos_noise).
                 bag_yaw_range: Tuple[float, float] = (np.radians(-15), np.radians(105)),
                 hand_yaw_noise: float = 0.8,
                 hand_tilt_noise: float = 0.35,
                 hand_start_pos: Tuple[float, float, float] = (0.5, 0.0, 0.28),
                 seed: Optional[int] = None):
        self.image_size = image_size
        self.n_candies = n_candies
        self.candy_radius = candy_radius
        self.bag_size = np.array(bag_size)
        self.wall = wall
        self.max_episode_steps = max_episode_steps
        self.max_pos_step = max_pos_step
        self.max_rot_step = max_rot_step
        self.bag_pos_noise = bag_pos_noise
        self.bag_yaw_range = tuple(bag_yaw_range)
        # start hand orientation: home (pointing down) rotated about world z, then tilted off vertical
        self.hand_yaw_noise = hand_yaw_noise
        self.hand_tilt_noise = hand_tilt_noise
        self.hand_start_pos = np.array(hand_start_pos)  # TCP position at reset (low enough for the front camera)
        self.bag_home = np.array([0.5, 0.0, bag_size[2] / 2])

        self.model = self._build_model().compile()
        self.model.opt.timestep = 0.002
        self.n_substeps = int(round(control_dt / self.model.opt.timestep))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, image_size, image_size)

        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_dofadr = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.tcp_site = m.site("tcp").id
        self.bag_body = m.body("bag").id
        self.bag_qadr = m.jnt_qposadr[m.joint("bag_free").id]
        self.candy_bodies = np.array([m.body(f"candy{i}").id for i in range(n_candies)])
        self.candy_qadr = np.array([m.jnt_qposadr[m.joint(f"candy{i}_free").id] for i in range(n_candies)])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "wrist_image": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (8,), np.float32),  # eef pos (3), quat (4), gripper width (1)
        })
        self.np_random = np.random.default_rng(seed)

    # ------------------------------------------------------------------ scene
    def _build_model(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(PANDA_XML))
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.noslip_iterations = 3  # keeps the pinch grasp from slipping
        world = spec.worldbody

        world.add_light(pos=[0.5, 0, 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7],
                        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05],
                       rgba=[0.85, 0.82, 0.75, 1], friction=[1.0, 0.005, 0.0001])
        # low and close: the bag fills ~45% more pixels than from further back and the camera sees into the open end
        world.add_camera(name="front", pos=[0.88, -0.36, 0.30], fovy=52,
                         xyaxes=_look_at_xyaxes([0.88, -0.36, 0.30], [0.5, 0.0, 0.17]))

        hand = spec.body("hand")
        hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005] * 3, rgba=[1, 0, 0, 0])
        # looks down the hand z-axis (toward the fingers), tilted 25 deg back toward the hand centre
        tilt = np.deg2rad(25)
        hand.add_camera(name="wrist", pos=[0.08, 0, 0.045], fovy=75,
                        xyaxes=[0, 1, 0, np.cos(tilt), 0, np.sin(tilt)])

        self._add_bag(world)
        self._add_candies(world)
        return spec

    def _add_bag(self, world) -> None:
        """Rigid open-ended box lying on its side: closed end at +y, open end at -y."""
        w, l, h = self.bag_size
        t = self.wall
        bag = world.add_body(name="bag", pos=self.bag_home)
        bag.add_freejoint(name="bag_free")
        walls = {
            "bottom": ([0, 0, -h / 2 + t / 2], [w / 2, l / 2, t / 2]),
            "top": ([0, 0, h / 2 - t / 2], [w / 2, l / 2, t / 2]),
            "left": ([-w / 2 + t / 2, 0, 0], [t / 2, l / 2, h / 2]),
            "right": ([w / 2 - t / 2, 0, 0], [t / 2, l / 2, h / 2]),
            "closed_end": ([0, l / 2 - t / 2, 0], [w / 2, t / 2, h / 2]),
        }
        for name, (pos, half) in walls.items():
            rgba = [0.75, 0.1, 0.1, 1] if name != "closed_end" else [0.45, 0.05, 0.05, 1]
            bag.add_geom(name=f"bag_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half,
                         density=300, rgba=rgba, friction=[1.5, 0.005, 0.0001])
        # Visual-only liner on the inner faces, in a light contrasting colour (shaded grey by the top wall), so the open end shows as a
        # patch in the cameras (otherwise both ends of the red box look alike). No collisions, no mass.
        e = 0.0005  # liner half-thickness
        inner = np.array([w / 2 - t, l / 2 - t, h / 2 - t])  # half-extents of the cavity
        liners = {
            "bottom": ([0, -t / 2, -inner[2] + e], [inner[0], inner[1] + t / 2, e]),
            "top": ([0, -t / 2, inner[2] - e], [inner[0], inner[1] + t / 2, e]),
            "left": ([-inner[0] + e, -t / 2, 0], [e, inner[1] + t / 2, inner[2]]),
            "right": ([inner[0] - e, -t / 2, 0], [e, inner[1] + t / 2, inner[2]]),
            "closed_end": ([0, inner[1] - e, 0], [inner[0], e, inner[2]]),
        }
        for name, (pos, half) in liners.items():
            bag.add_geom(name=f"bag_liner_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half,
                         rgba=[0.95, 0.95, 0.85, 1], contype=0, conaffinity=0, density=0)

    def _add_candies(self, world) -> None:
        colors = [[0.9, 0.1, 0.1, 1], [1.0, 0.55, 0.0, 1], [1.0, 0.9, 0.1, 1],
                  [0.2, 0.7, 0.2, 1], [0.5, 0.2, 0.6, 1]]
        for i in range(self.n_candies):
            body = world.add_body(name=f"candy{i}", pos=[0, 0, -1])
            body.add_freejoint(name=f"candy{i}_free")
            # condim=6 enables rolling friction, otherwise spheres roll out of the open end on their own
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[self.candy_radius, 0, 0], mass=0.001,
                          rgba=colors[i % len(colors)], friction=[0.5, 0.005, 0.003], condim=6)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[self.arm_qadr] = self._random_start_qpos()
        d.ctrl[:7] = d.qpos[self.arm_qadr]
        d.ctrl[7] = 255  # gripper open

        bag_pos = self.bag_home.copy()
        bag_pos[:2] += self.np_random.uniform(-self.bag_pos_noise, self.bag_pos_noise, 2)
        yaw = self.np_random.uniform(*self.bag_yaw_range)
        bag_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        d.qpos[self.bag_qadr:self.bag_qadr + 3] = bag_pos
        d.qpos[self.bag_qadr + 3:self.bag_qadr + 7] = bag_quat
        for qadr, p in zip(self.candy_qadr, self._candy_spawn_positions(bag_pos, bag_quat)):
            d.qpos[qadr:qadr + 3] = p
            d.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, d)
        for _ in range(250):  # let candies settle (0.5 s)
            mujoco.mj_step(self.model, d)

        self.target_pos = d.site_xpos[self.tcp_site].copy()
        self.target_quat = self._site_quat()
        self.failure = False
        self.steps = 0
        return self._get_obs(), self._get_info()

    def _random_start_qpos(self, iters: int = 100) -> np.ndarray:
        """Joint angles putting the TCP at ``hand_start_pos`` with a randomly rotated orientation."""
        m, d = self.model, self.data
        d.qpos[self.arm_qadr] = HOME_QPOS
        mujoco.mj_kinematics(m, d)
        target_pos, home_quat = self.hand_start_pos, self._site_quat()

        yaw = self.np_random.uniform(-self.hand_yaw_noise, self.hand_yaw_noise)
        tilt = self.np_random.uniform(0, self.hand_tilt_noise)
        tilt_dir = self.np_random.uniform(0, 2 * np.pi)
        q_yaw, q_tilt, tmp, target_quat = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0]), yaw)
        mujoco.mju_axisAngle2Quat(q_tilt, np.array([np.cos(tilt_dir), np.sin(tilt_dir), 0.0]), tilt)
        mujoco.mju_mulQuat(tmp, q_yaw, home_quat)
        mujoco.mju_mulQuat(target_quat, q_tilt, tmp)

        for _ in range(iters):  # kinematic IK, no physics stepping
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            d.qpos[self.arm_qadr] = self._ik_joint_targets(target_pos, target_quat)
        return d.qpos[self.arm_qadr].copy()

    def _candy_spawn_positions(self, bag_pos, bag_quat) -> np.ndarray:
        """Grid inside the bag, in the closed half so nothing starts near the opening."""
        w, l, h = self.bag_size
        r, t = self.candy_radius, self.wall
        gap = 2 * r + 0.001
        xs = np.arange(-w / 2 + t + r + 0.001, w / 2 - t - r, gap)
        zs = np.arange(-h / 2 + t + r + 0.001, h / 2 - t - r, gap)
        ys = np.arange(l / 2 - t - r - 0.001, -l / 2 * 0.2, -gap)
        grid = np.array([[x, y, z] for y in ys for z in zs for x in xs])
        if len(grid) < self.n_candies:
            raise ValueError(f"bag fits {len(grid)} candies, asked for {self.n_candies}")
        local = grid[: self.n_candies] + self.np_random.uniform(-0.0005, 0.0005, (self.n_candies, 3))
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, bag_quat)
        return bag_pos + local @ rot.reshape(3, 3).T

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.target_pos = self.target_pos + action[:3] * self.max_pos_step
        rotvec = action[3:6] * self.max_rot_step
        angle = np.linalg.norm(rotvec)
        if angle > 1e-9:
            dq, new_quat = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, rotvec / angle, angle)
            mujoco.mju_mulQuat(new_quat, dq, self.target_quat)
            self.target_quat = new_quat / np.linalg.norm(new_quat)
        self.data.ctrl[7] = 0.0 if action[6] > 0 else 255.0

        for i in range(self.n_substeps):
            if i % 5 == 0:
                self.data.ctrl[:7] = self._ik_joint_targets(self.target_pos, self.target_quat)
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        info = self._get_info()
        self.failure = self.failure or info["n_spilled"] > 0
        info["failure"] = self.failure
        truncated = self.steps >= self.max_episode_steps
        return self._get_obs(), 0.0, False, truncated, info

    def _ik_joint_targets(self, target_pos, target_quat, damping: float = 1e-2) -> np.ndarray:
        """Damped least-squares IK step from the TCP pose error to joint position targets."""
        m, d = self.model, self.data
        err = np.zeros(6)
        err[:3] = target_pos - d.site_xpos[self.tcp_site]
        mujoco.mju_subQuat(err[3:], target_quat, self._site_quat())
        err[3:] = d.site_xmat[self.tcp_site].reshape(3, 3) @ err[3:]  # local -> world frame (matches jacr)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp_site)
        J = np.vstack([jacp, jacr])[:, self.arm_dofadr]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        lo, hi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)]].T
        return np.clip(d.qpos[self.arm_qadr] + dq, lo, hi)

    def _site_quat(self) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.tcp_site])
        return q

    # ------------------------------------------------------------------ obs / info
    def candies_in_bag_frame(self) -> np.ndarray:
        d = self.data
        rot = d.xmat[self.bag_body].reshape(3, 3)
        return (d.xpos[self.candy_bodies] - d.xpos[self.bag_body]) @ rot

    def _get_info(self) -> Dict[str, Any]:
        local = self.candies_in_bag_frame()
        outside = np.any(np.abs(local) > self.bag_size / 2 + 0.002, axis=1)
        return {
            "n_spilled": int(outside.sum()),
            "failure": getattr(self, "failure", False),
            "bag_pos": self.data.xpos[self.bag_body].copy(),
            "bag_quat": self.data.xquat[self.bag_body].copy(),
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        d = self.data
        finger_width = d.qpos[self.model.jnt_qposadr[self.model.joint("finger_joint1").id]] * 2
        state = np.concatenate([d.site_xpos[self.tcp_site], self._site_quat(), [finger_width]])
        return {
            "image": self.render("front"),
            "wrist_image": self.render("wrist"),
            "state": state.astype(np.float32),
        }

    def render(self, camera: str = "front") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()

    # ------------------------------------------------------------------ helpers for scripted experts
    def bag_point(self, local: np.ndarray) -> np.ndarray:
        """World position of a point given in the bag frame."""
        rot = self.data.xmat[self.bag_body].reshape(3, 3)
        return self.data.xpos[self.bag_body] + rot @ local

    def bag_yaw(self) -> float:
        q = self.data.xquat[self.bag_body]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
