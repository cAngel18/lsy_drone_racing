"""Train the racing policy of [rl_racer][lsy_drone_racing.control.rl_racer] with PPO.

Rollouts and updates are one jitted JAX function, so training only needs the packages of the default
environment. Three choices make it practical on a CPU and reliable on this track:

* **Analytic collision check.** The MJX contact check takes about 80 % of the environment step. For
  training it is replaced by a separating-axis test on the same boxes. Checkpoints are selected by
  evaluations on the unmodified environment.
* **Progress along a reference line** as the main reward (MPCC-like, Romero et al., T-RO 2022). The
  line only measures progress and is not tracked, the policy is free to cut corners. Unlike the
  distance to the next gate, it does not lure the drone to the wrong side of a gate. Progress is
  capped at the next gate until that gate has been passed.
* **Reference state initialisation.** Part of the episodes start somewhere along the line, so later
  gates are practised before the drone can reach them from the start.

The actor sees only the race observation. The critic additionally sees the true track and mass
(asymmetric actor-critic), which is available in simulation and only used during training.

Run a stage as:

    $ python lsy_drone_racing/control/train_rl_racer.py --total_steps 60_000_000 --run_name stage1

Checkpoints go to ``saves/rl_racer/<run_name>``. The shipped policy took six stages (351 M steps),
each continuing from the previous checkpoint with ``--init_policy``. Later stages trade reliability
for speed through ``--crash_bootstrap``, which unlike a lower ``--gamma`` keeps the planning
horizon. Overrides of `Args`, and the success rate and mean lap on the unmodified environment:

1. 60 M steps, no overrides: 72 %, 4.84 s.
2. 80 M steps, lr 2e-4, lr_final 2e-5, log_std_min -3, entropy_coef 1e-3, rsi_start 0.3,
   rsi_end 0.15, rsi_speed 4, gamma 0.98: 77 %, 4.33 s.
3. 56 M steps, lr 2e-4, lr_final 2e-5, log_std_min -3.5, entropy_coef 5e-4, rsi_start 0.2,
   rsi_end 0.1, rsi_speed 4, gate_center_bonus 0.3, gamma 0.97: 64 %, 4.25 s.
4. 60 M steps, as stage 3 but lr_final 3e-5, gamma 0.99, crash_bootstrap 0.85: 69 %, 4.15 s.
5. 62 M steps, as stage 4 but lr 1.5e-4, lr_final 2e-5, rsi_start 0.15, crash_bootstrap 0.95,
   gate_center_bonus 0.1: 71 %, 4.03 s.
6. 33 M steps, as stage 5 but rsi_start 0.12, rsi_end 0.08, ang_vel 2e-5, max_tilt 1.5:
   61 %, 3.91 s.

Training is stochastic, so a rerun will land near these values, not on them.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
import jax
import jax.numpy as jp
import numpy as np
import optax
from crazyflow.dynamics import load_params as load_dynamics_params
from flax import struct
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.rl_racer import (
    G,
    action_to_command,
    mass_estimate_init,
    mass_estimate_update,
    mass_ratio,
    mlp_forward,
    policy_observation,
    quat_to_matrix,
)
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import obs as race_obs
from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.envs.race_core import EnvData

ROOT = Path(__file__).parents[2]
CHECKPOINT_DIR = ROOT / "saves" / "rl_racer"

# Collision geometry, from the gate, obstacle and cf21B_500 models
DRONE_HALF = (0.07, 0.07, 0.02)
GATE_BOX_POS = (
    (0.0, 0.0, 0.28),
    (0.0, 0.0, -0.28),
    (0.0, -0.28, 0.0),
    (0.0, 0.28, 0.0),
    (0.0, 0.0, -0.86),
)
GATE_BOX_HALF = (
    (0.01, 0.36, 0.08),
    (0.01, 0.36, 0.08),
    (0.01, 0.08, 0.36),
    (0.01, 0.08, 0.36),
    (0.05, 0.05, 0.5),
)
OBSTACLE_HALF = (0.015, 0.015, 0.815)
OBSTACLE_CENTER_Z = -0.8  # Relative to the reported obstacle position at the top
# Line through the nominal track: (kind, index, offset). Points were optimised for the MPCC racer.
GATE_TANGENTS = (0.988, 1.213, None, 1.143, 1.573)  # Hermite tangent magnitude per gate pass
LINE_POINTS = (
    ("start", 0, (0.0, 0.0, 0.0)),
    ("obstacle", 0, (-0.382, -0.234, 0.464)),
    ("gate", 0, None),
    ("obstacle", 1, (0.229, -0.285, 0.915)),
    ("gate", 1, None),
    ("point", 0, (0.485, 0.75, 1.155)),
    ("point", 0, (-0.443, -0.037, 0.699)),
    ("gate", 2, None),
    ("obstacle", 2, (0.118, 0.265, 1.279)),
    ("obstacle", 3, (0.045, 0.213, 1.201)),
    ("gate", 3, None),
    ("obstacle", 1, (0.137, -0.302, 1.205)),
    ("gate", 1, None),
    ("point", 0, (0.5, 1.25, 1.2)),
)
LINE_DS = 0.05  # [m] resolution of the sampled line
LINE_WINDOW = (0.3, 0.5)  # [m] search window (back, forward) of the progress projection
RSI_GATE_CLEARANCE = 0.3  # [m] no reference state starts this close to a gate on the line


@dataclass
class Args:
    """Training settings."""

    seed: int = 0
    total_steps: int = 100_000_000
    num_envs: int = 2048
    num_steps: int = 32
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    crash_bootstrap: float = 0.0  # share of the value surviving a crash, sets the risk tolerance
    lr: float = 3e-4
    lr_final: float = 3e-5
    epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0  # applied separately to the actor and the critic
    value_scale: float = 10.0  # the critic predicts returns divided by this value
    log_std_init: float = -1.2
    log_std_min: float = -2.3
    hidden: tuple[int, ...] = (128, 128, 128)
    # Reward
    progress: float = 5.0  # per metre of the line
    speed_bonus: float = 0.0  # relative increase of the progress reward at 3 m/s along the line
    gate_bonus: float = 1.0  # per gate pass
    gate_center_bonus: float = 1.0  # additional, for passing through the centre of the gate
    finish_bonus: float = 5.0
    crash: float = 2.0
    time_penalty: float = 0.0  # per step
    action_rate: float = 0.01  # on the squared action difference
    ang_vel: float = 1e-4  # on the squared angular velocity
    # Episodes
    max_episode_steps: int = 500
    drone_inflate: float = 0.005  # [m] added to the drone box in the training collision check
    off_line_limit: float = 1.5  # [m] episodes end this far from the line
    rsi_start: float = 0.6  # probability of a reference state start at the beginning of training
    rsi_end: float = 0.2  # and at the end
    rsi_speed: float = 3.0  # [m/s] maximum speed of reference states
    max_tilt: float = 1.2  # [rad] roll and pitch command at the action limit
    # Evaluation
    eval_every: int = 50  # iterations
    eval_envs: int = 512
    eval_steps: int = 500
    min_success: float = 0.65  # checkpoints below this success rate are penalised
    init_policy: str = ""  # checkpoint to continue from
    run_name: str = "run"


class FastContactRaceEnv(VecDroneRaceEnv):
    """Race environment with an analytic collision check on the geometry of the MJX contact check.

    The drone box can be inflated with the class attribute ``drone_inflate``, which also leaves the
    policy a margin to the true collision geometry.
    """

    drone_inflate = 0.0

    def build_contact_check_fn(self) -> Any:
        """Build a separating-axis test of the drone box against all gate and obstacle boxes."""
        drone_half = jp.array(DRONE_HALF, dtype=jp.float32) + self.drone_inflate
        box_pos = jp.array(GATE_BOX_POS, dtype=jp.float32)
        box_half = jp.array(GATE_BOX_HALF, dtype=jp.float32)
        obstacle_half = jp.array(OBSTACLE_HALF, dtype=jp.float32)

        def check_contacts(data: EnvData) -> Array:
            drone_pos = data.sim_data.states.pos[:, 0][:, None, None]  # (n, 1, 1, 3)
            drone_rot = quat_to_matrix(jp, data.sim_data.states.quat[:, 0])[:, None, None]
            gate_rot = quat_to_matrix(jp, data.gates_quat)  # (n, g, 3, 3)
            gate_box_pos = (
                jp.einsum("ngij,bj->ngbi", gate_rot, box_pos) + data.gates_pos[:, :, None]
            )
            gate_box_rot = gate_rot[:, :, None]  # (n, g, 1, 3, 3)
            hit_gates = boxes_overlap(
                drone_pos, drone_rot, drone_half, gate_box_pos, gate_box_rot, box_half
            )
            obstacle_pos = data.obstacles_pos + jp.array([0.0, 0.0, OBSTACLE_CENTER_Z])
            hit_obstacles = boxes_overlap(
                drone_pos[:, 0], drone_rot[:, 0], drone_half, obstacle_pos, jp.eye(3), obstacle_half
            )
            return (jp.any(hit_gates, axis=(1, 2)) | jp.any(hit_obstacles, axis=1))[:, None]

        return check_contacts


def boxes_overlap(
    pos_a: Array, rot_a: Array, half_a: Array, pos_b: Array, rot_b: Array, half_b: Array
) -> Array:
    """Separating-axis test between oriented boxes, broadcasting over the leading dimensions.

    Args:
        pos_a: Centers of the first boxes (..., 3).
        rot_a: Rotation matrices of the first boxes (..., 3, 3).
        half_a: Half sizes of the first boxes (..., 3).
        pos_b: Centers of the second boxes (..., 3).
        rot_b: Rotation matrices of the second boxes (..., 3, 3).
        half_b: Half sizes of the second boxes (..., 3).

    Returns:
        True where the boxes overlap.
    """
    axes_a = jp.swapaxes(rot_a, -1, -2)  # Rows are the box axes
    axes_b = jp.swapaxes(rot_b, -1, -2)
    axes_a, axes_b = jp.broadcast_arrays(axes_a, axes_b)
    cross = jp.cross(axes_a[..., :, None, :], axes_b[..., None, :, :])
    axes = jp.concatenate([axes_a, axes_b, cross.reshape(cross.shape[:-3] + (9, 3))], axis=-2)
    dist = jp.abs(jp.sum((pos_b - pos_a)[..., None, :] * axes, axis=-1))
    radius_a = jp.sum(jp.abs(axes @ jp.swapaxes(axes_a, -1, -2)) * half_a[..., None, :], axis=-1)
    radius_b = jp.sum(jp.abs(axes @ jp.swapaxes(axes_b, -1, -2)) * half_b[..., None, :], axis=-1)
    return ~jp.any(dist > radius_a + radius_b, axis=-1)


def make_env(cls: type[VecDroneRaceEnv], config: Any, args: Args, n: int, seed: int) -> Any:
    """Create a vectorized race environment without autoreset (resets happen in the same step)."""
    env = cls(
        num_envs=n,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=seed,
        max_episode_steps=args.max_episode_steps,
    )
    env.settings = env.settings.replace(autoreset=False)
    env._step = env.build_step_fn()
    return env


def build_reference_line(config: Any) -> dict[str, np.ndarray]:
    """Sample the reference line through the nominal track.

    Args:
        config: The race configuration.

    Returns:
        Line samples ``pos`` and ``tangent``, arc length ``s``, the sample index ``gate_idx`` of
        each gate pass, the number of passed gates ``passed`` per sample, and which gates and
        obstacles have been in sensor range per sample.
    """
    track = config.env.track
    gates = np.array([g["pos"] for g in track.gates], dtype=float)
    gate_rpy = np.array([g["rpy"] for g in track.gates], dtype=float)
    obstacles = np.array([o["pos"] for o in track.obstacles], dtype=float)
    start = np.array(track.drones[0]["pos"], dtype=float)

    points, gate_of_point = [], []
    for kind, idx, offset in LINE_POINTS:
        gate_of_point.append(idx if kind == "gate" else -1)
        if kind == "start":
            points.append(start)
        elif kind == "gate":
            points.append(gates[idx])
        elif kind == "obstacle":
            points.append(np.array([*(obstacles[idx][:2] + offset[:2]), offset[2]]))
        else:
            points.append(np.array(offset))
    points = np.array(points)

    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    tangents = np.empty_like(points)
    tangents[1:-1] = (points[2:] - points[:-2]) / (u[2:] - u[:-2])[:, None]
    tangents[0] = (points[1] - points[0]) / (u[1] - u[0])
    tangents[-1] = (points[-1] - points[-2]) / (u[-1] - u[-2])
    gate_pass = 0
    for i, gate in enumerate(gate_of_point):
        if gate < 0:
            continue
        magnitude = GATE_TANGENTS[gate_pass]
        gate_pass += 1
        if magnitude is not None:
            normal = R.from_euler("xyz", gate_rpy[gate]).apply([1.0, 0.0, 0.0])
            normal[2] = 0.0
            tangents[i] = magnitude * normal / np.linalg.norm(normal)

    spline = CubicHermiteSpline(u, points, tangents)
    fine = spline(np.linspace(0.0, u[-1], 5000))
    s_fine = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))])
    s = np.arange(0.0, s_fine[-1], LINE_DS)
    pos = np.stack([np.interp(s, s_fine, fine[:, i]) for i in range(3)], axis=1)
    tangent = np.gradient(pos, s, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    sequence = np.abs(np.array(track.gate_order)) - 1
    gate_idx, lo = [], 0
    for gate in sequence:
        window = pos[lo : lo + int(4.5 / LINE_DS)]
        gate_idx.append(lo + int(np.argmin(np.linalg.norm(window - gates[gate], axis=1))))
        lo = gate_idx[-1] + 10
    gate_idx = np.array(gate_idx)
    passed = np.searchsorted(gate_idx, np.arange(len(s)), side="left")

    sensor = config.env.sensor_range
    d_gates = np.linalg.norm(pos[:, None, :2] - gates[None, :, :2], axis=-1) < sensor
    d_obstacles = np.linalg.norm(pos[:, None, :2] - obstacles[None, :, :2], axis=-1) < sensor
    near_gate = np.min(np.abs(np.arange(len(s))[:, None] - gate_idx[None]), axis=1) * LINE_DS
    rsi_idx = np.flatnonzero((near_gate > RSI_GATE_CLEARANCE) & (np.arange(len(s)) < gate_idx[-1]))
    return {
        "pos": pos.astype(np.float32),
        "tangent": tangent.astype(np.float32),
        "s": s.astype(np.float32),
        "gate_idx": gate_idx,
        "passed": passed,
        "gates_seen": np.logical_or.accumulate(d_gates, axis=0),
        "obstacles_seen": np.logical_or.accumulate(d_obstacles, axis=0),
        "rsi_idx": rsi_idx,
    }


@struct.dataclass
class RaceState:
    """The state of the training task on top of the environment data."""

    env: EnvData
    line_idx: Array  # Index of the projection on the line (n,)
    line_s: Array  # Refined progress along the line (n,)
    act_prev: Array  # Last action (n, 3)
    act_prev2: Array  # Action before the last one (n, 3)
    thrust_cmd: Array  # Last commanded thrust (n,)
    est: dict[str, Array]  # Mass estimator
    from_start: Array  # True for episodes that started at the regular start (n,)
    rng: Array


class RaceTask:
    """Reward, resets and observations of the race on top of a vectorized race environment."""

    def __init__(self, env: Any, config: Any, args: Args, limits: dict[str, float]):
        """Build the task.

        Args:
            env: The vectorized race environment.
            config: The race configuration.
            args: The training settings.
            limits: The action limits.
        """
        self.env, self.args, self.limits = env, args, limits
        self.n = env.num_envs
        self.dt = 1.0 / config.env.freq
        line = build_reference_line(config)
        self.line = {k: jp.asarray(v) for k, v in line.items()}
        self.window = (int(LINE_WINDOW[0] / LINE_DS), int(LINE_WINDOW[1] / LINE_DS))
        self.pos_low = jp.array(config.env.track.safety_limits.pos_limit_low, dtype=jp.float32)
        self.pos_high = jp.array(config.env.track.safety_limits.pos_limit_high, dtype=jp.float32)
        self.gate_sequence = jp.asarray(env.data.gate_sequence)
        self.mass_nominal = float(limits["mass_nominal"])
        # Rotor speed at hover, which reference states start from
        params = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        c = np.asarray(params["rpm2thrust"], dtype=float)
        hover = self.mass_nominal * G / 4.0
        self.rotor_hover = float(np.max(np.roots([c[2], c[1], c[0] - hover]).real))

    def init(self, rng: Array) -> RaceState:
        """Create the initial state with all episodes started from the regular start."""
        data, _ = self.env._reset(self.env.data)
        return RaceState(
            env=data,
            line_idx=jp.zeros(self.n, dtype=jp.int32),
            line_s=jp.zeros(self.n, dtype=jp.float32),
            act_prev=jp.zeros((self.n, 3), dtype=jp.float32),
            act_prev2=jp.zeros((self.n, 3), dtype=jp.float32),
            thrust_cmd=jp.full((self.n,), self.mass_nominal * G, dtype=jp.float32),
            est=mass_estimate_init(jp, self.n, self.mass_nominal),
            from_start=jp.ones(self.n, dtype=bool),
            rng=rng,
        )

    def reset(self, state: RaceState, mask: Array, p_rsi: Array) -> RaceState:
        """Reset the masked episodes, a fraction p_rsi of them to reference states on the line."""
        args, line = self.args, self.line
        data, _ = self.env._reset(state.env, None, mask)
        rng, *keys = jax.random.split(state.rng, 8)

        rsi = mask & (jax.random.uniform(keys[0], (self.n,)) < p_rsi)
        choice = jax.random.randint(keys[1], (self.n,), 0, line["rsi_idx"].shape[0])
        idx = line["rsi_idx"][choice]
        noise = jax.random.uniform(keys[2], (self.n, 3), minval=-0.1, maxval=0.1)
        rsi_pos = line["pos"][idx] + noise
        speed = jax.random.uniform(keys[3], (self.n, 1), minval=0.3, maxval=args.rsi_speed)
        rsi_vel = line["tangent"][idx] * speed
        half_tilt = jax.random.uniform(keys[4], (self.n, 2), minval=-0.2, maxval=0.2)
        sr, cr, sp, cp = (
            jp.sin(half_tilt[:, 0]),
            jp.cos(half_tilt[:, 0]),
            jp.sin(half_tilt[:, 1]),
            jp.cos(half_tilt[:, 1]),
        )
        rsi_quat = jp.stack([sr * cp, cr * sp, -sr * sp, cr * cp], axis=-1)
        rotor = self.rotor_hover * jax.random.uniform(keys[5], (self.n, 1), minval=0.85, maxval=1.2)

        def select(rsi_value: Array, value: Array) -> Array:
            return jp.where(rsi.reshape(rsi.shape + (1,) * (value.ndim - 1)), rsi_value, value)

        states = data.sim_data.states
        pos = select(rsi_pos[:, None], states.pos)
        vel = select(rsi_vel[:, None], states.vel)
        states = states.replace(
            pos=pos,
            vel=vel,
            quat=select(rsi_quat[:, None], states.quat),
            rotor_vel=select(jp.tile(rotor[:, None], (1, 1, 4)), states.rotor_vel),
        )
        data = data.replace(
            sim_data=data.sim_data.replace(states=states),
            n_gates_passed=select(line["passed"][idx][:, None], data.n_gates_passed),
            gates_visited=select(
                line["gates_seen"][idx][:, None] | data.gates_visited, data.gates_visited
            ),
            obstacles_visited=select(
                line["obstacles_seen"][idx][:, None] | data.obstacles_visited,
                data.obstacles_visited,
            ),
            last_drone_pos=select(pos, data.last_drone_pos),
            takeoff_pos=select(pos, data.takeoff_pos),
        )

        def on_reset(reset_value: Array, value: Array) -> Array:
            return jp.where(mask.reshape(mask.shape + (1,) * (value.ndim - 1)), reset_value, value)

        fresh = mass_estimate_init(jp, self.n, self.mass_nominal)
        return state.replace(
            env=data,
            line_idx=on_reset(jp.where(rsi, idx, 0), state.line_idx),
            line_s=on_reset(jp.where(rsi, line["s"][idx], 0.0), state.line_s),
            act_prev=on_reset(jp.zeros_like(state.act_prev), state.act_prev),
            act_prev2=on_reset(jp.zeros_like(state.act_prev2), state.act_prev2),
            thrust_cmd=on_reset(
                jp.full_like(state.thrust_cmd, self.mass_nominal * G), state.thrust_cmd
            ),
            est={k: on_reset(fresh[k], state.est[k]) for k in fresh},
            from_start=on_reset(~rsi, state.from_start),
            rng=rng,
        )

    def _project(self, state: RaceState, pos: Array, passed: Array) -> tuple[Array, Array, Array]:
        """Project positions on the part of the line that is reachable before the next gate."""
        line = self.line
        n_line = line["pos"].shape[0]
        cap = jp.where(
            passed < line["gate_idx"].shape[0],
            line["gate_idx"][jp.minimum(passed, line["gate_idx"].shape[0] - 1)],
            n_line - 1,
        )
        lo = state.line_idx - self.window[0]
        hi = jp.minimum(state.line_idx + self.window[1], cap)
        dist2 = jp.sum((pos[:, None] - line["pos"][None]) ** 2, axis=-1)
        index = jp.arange(n_line)[None]
        dist2 = jp.where((index >= lo[:, None]) & (index <= hi[:, None]), dist2, jp.inf)
        idx = jp.argmin(dist2, axis=1)
        # Refine within the sample so that the progress does not jump with the resolution
        along = jp.sum((pos - line["pos"][idx]) * line["tangent"][idx], axis=-1)
        s = line["s"][idx] + jp.clip(along, -LINE_DS, LINE_DS)
        s = jp.minimum(s, line["s"][cap])
        return idx, s, jp.sqrt(jp.min(dist2, axis=1))

    def step(self, state: RaceState, action: Array) -> tuple[RaceState, Array, Array, dict]:
        """Step the race without resetting.

        Args:
            state: The current state.
            action: Policy actions in [-1, 1] (n, 3).

        Returns:
            The next state, the reward, the episode end flag and episode statistics.
        """
        args = self.args
        command = action_to_command(jp, action, self.limits)
        passed_before = state.env.n_gates_passed[:, 0]
        vel_before = state.env.sim_data.states.vel[:, 0]
        data, (_, _, disabled, truncated, _) = self.env._step(state.env, command)
        disabled, truncated = disabled[:, 0], truncated[:, 0]

        states = data.sim_data.states
        pos = states.pos[:, 0]
        passed = data.n_gates_passed[:, 0]
        passed_gate = passed > passed_before
        finished = passed >= self.gate_sequence.shape[0]

        idx, s, off_line = self._project(state, pos, passed)
        outside = jp.any((pos < self.pos_low) | (pos > self.pos_high), axis=-1)
        crashed = (disabled | (off_line > args.off_line_limit) | outside) & ~finished
        progress = jp.clip(s - state.line_s, -0.25, 0.25)

        # Gate-centre term, from the crossing point in the frame of the gate that was just passed
        row = jp.arange(self.n)
        gate = self.gate_sequence[jp.minimum(passed_before, self.gate_sequence.shape[0] - 1)]
        gate_rot = quat_to_matrix(jp, data.gates_quat[row, gate])
        local = jp.sum(gate_rot * (pos - data.gates_pos[row, gate])[..., None], axis=-2)
        offset = jp.maximum(jp.abs(local[:, 1]), jp.abs(local[:, 2]))
        centered = jp.clip(1.0 - (offset / 0.2) ** 2, 0.0, 1.0)

        speed_factor = 1.0 + args.speed_bonus * progress / (3.0 * self.dt)
        reward = args.progress * progress * speed_factor * ~crashed
        reward += passed_gate * (args.gate_bonus + args.gate_center_bonus * centered)
        reward += finished * args.finish_bonus
        reward -= crashed * args.crash
        reward -= args.time_penalty
        reward -= args.action_rate * jp.sum((action - state.act_prev) ** 2, axis=-1)
        reward -= args.ang_vel * jp.sum(states.ang_vel[:, 0] ** 2, axis=-1)

        done = crashed | finished | truncated
        est = mass_estimate_update(
            jp,
            state.est,
            pos,
            states.quat[:, 0],
            states.vel[:, 0],
            vel_before,
            state.thrust_cmd,
            self.limits,
            self.dt,
        )
        next_state = state.replace(
            env=data,
            line_idx=idx,
            line_s=s,
            act_prev=jp.clip(action, -1.0, 1.0),
            act_prev2=state.act_prev,
            thrust_cmd=command[:, 3],
            est=est,
        )
        stats = {
            "finished": finished,
            "crashed": crashed,
            "truncated": truncated & ~finished & ~crashed,
            "passed": passed,
            "time": data.steps * self.dt - self.dt,
            "s": s,
        }
        return next_state, reward, done, stats

    def observe(self, state: RaceState) -> tuple[Array, Array]:
        """The policy input and the critic input of the current state."""
        raw = race_obs(state.env)
        race = {k: v[:, 0] for k, v in raw.items()}
        ratio = mass_ratio(jp, state.est, self.mass_nominal)
        policy_in = policy_observation(
            jp, race, self.gate_sequence, state.act_prev, state.act_prev2, ratio
        )
        # Privileged critic input: the true poses behind the observed ones and the true mass
        data = state.env
        row = jp.arange(self.n)
        pos = race["pos"]
        passed = race["n_gates_passed"]
        n_passes = self.gate_sequence.shape[0]
        gate = self.gate_sequence[jp.minimum(passed, n_passes - 1)]
        next_gate = self.gate_sequence[jp.minimum(passed + 1, n_passes - 1)]
        mass = data.sim_data.params.mass[:, 0, 0]
        privileged = [
            (data.gates_pos[row, gate] - pos) / 2.0,
            (data.gates_pos[row, next_gate] - pos) / 2.0,
            (data.obstacles_pos[..., :2] - pos[:, None, :2]).reshape(self.n, -1) / 2.0,
            ((mass / self.mass_nominal - 1.0) / 0.12)[:, None],
            (state.line_s / self.line["s"][-1])[:, None],
        ]
        return policy_in, jp.concatenate([policy_in, *privileged], axis=-1)


def init_mlp(key: Array, sizes: list[int], final_scale: float) -> list[tuple[Array, Array]]:
    """Create MLP layers with orthogonal initialisation."""
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = final_scale if i == len(sizes) - 2 else np.sqrt(2)
        w = jax.nn.initializers.orthogonal(scale)(k, (n_in, n_out), jp.float32)
        layers.append((w, jp.zeros(n_out, dtype=jp.float32)))
    return layers


def critic_forward(layers: list[tuple[Array, Array]], x: Array) -> Array:
    """Forward pass of the critic."""
    for w, b in layers[:-1]:
        x = jax.nn.elu(x @ w + b)
    w, b = layers[-1]
    return (x @ w + b)[..., 0]


def gaussian_log_prob(action: Array, mean: Array, log_std: Array) -> Array:
    """Log probability of a diagonal Gaussian."""
    z = (action - mean) * jp.exp(-log_std)
    return jp.sum(-0.5 * z * z - log_std - 0.5 * jp.log(2 * jp.pi), axis=-1)


def clip_global_norm(tree: Any, max_norm: float) -> Any:
    """Scale a gradient tree down to a maximum global norm."""
    factor = jp.minimum(1.0, max_norm / (optax.global_norm(tree) + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * factor, tree)


def make_train_iter(task: RaceTask, args: Args, optimizer: Any) -> Any:
    """Build the jitted function running one rollout and one PPO update."""
    batch = args.num_envs * args.num_steps
    minibatch = batch // args.minibatches

    def rollout(params: dict, state: RaceState, p_rsi: Array) -> tuple:
        def body(state: RaceState, _: None) -> tuple:
            obs_pi, obs_vf = task.observe(state)
            mean = mlp_forward(jp, params["actor"], obs_pi)
            log_std = jp.maximum(params["log_std"], args.log_std_min)
            rng, key = jax.random.split(state.rng)
            action = mean + jp.exp(log_std) * jax.random.normal(key, mean.shape)
            logp = gaussian_log_prob(action, mean, log_std)
            value = args.value_scale * critic_forward(params["critic"], obs_vf)
            next_state, reward, done, stats = task.step(state.replace(rng=rng), action)
            from_start = state.from_start
            next_state = task.reset(next_state, done, p_rsi)
            transition = {
                "obs_pi": obs_pi,
                "obs_vf": obs_vf,
                "action": action,
                "logp": logp,
                "value": value,
                "reward": reward,
                "done": done,
                "crashed": stats["crashed"],
            }
            episode = {
                "ended": done & from_start,
                "finished": stats["finished"] & from_start,
                "crashed": stats["crashed"] & from_start,
                "time": stats["time"],
                "passed": stats["passed"],
                "reward": reward,
            }
            return next_state, (transition, episode)

        return jax.lax.scan(body, state, None, length=args.num_steps)

    def gae(transitions: dict, last_value: Array) -> tuple[Array, Array]:
        def body(carry: tuple, step: tuple) -> tuple:
            advantage, next_value = carry
            reward, done, crashed, value = step
            nonterminal = 1.0 - done
            # A crash ends the episode, but keeps a share of the value. Unlike a lower discount,
            # this makes the policy less risk averse without shortening its planning horizon.
            target = next_value * nonterminal + args.crash_bootstrap * crashed * value
            delta = reward + args.gamma * target - value
            advantage = delta + args.gamma * args.gae_lambda * nonterminal * advantage
            return (advantage, value), advantage

        steps = (
            transitions["reward"],
            transitions["done"].astype(jp.float32),
            transitions["crashed"].astype(jp.float32),
            transitions["value"],
        )
        _, advantages = jax.lax.scan(
            body, (jp.zeros_like(last_value), last_value), steps, reverse=True
        )
        return advantages, advantages + transitions["value"]

    def loss_fn(params: dict, mb: dict) -> tuple:
        mean = mlp_forward(jp, params["actor"], mb["obs_pi"])
        log_std = jp.maximum(params["log_std"], args.log_std_min)
        logp = gaussian_log_prob(mb["action"], mean, log_std)
        ratio = jp.exp(logp - mb["logp"])
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        policy_loss = jp.maximum(
            -adv * ratio, -adv * jp.clip(ratio, 1 - args.clip, 1 + args.clip)
        ).mean()
        value = critic_forward(params["critic"], mb["obs_vf"])
        value_loss = 0.5 * jp.mean((value - mb["ret"] / args.value_scale) ** 2)
        entropy = jp.sum(log_std + 0.5 * jp.log(2 * jp.pi * jp.e))
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
        kl = jp.mean(mb["logp"] - logp)
        return loss, {"policy_loss": policy_loss, "value_loss": value_loss, "kl": kl}

    def update(params: dict, opt_state: Any, data: dict, key: Array) -> tuple:
        def epoch(carry: tuple, key: Array) -> tuple:
            params, opt_state = carry
            perm = jax.random.permutation(key, batch)[: minibatch * args.minibatches]
            idx = perm.reshape(args.minibatches, minibatch)

            def step(carry: tuple, mb_idx: Array) -> tuple:
                params, opt_state = carry
                mb = {k: v[mb_idx] for k, v in data.items()}
                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                grads = grads | {
                    "actor": clip_global_norm(grads["actor"], args.max_grad_norm),
                    "critic": clip_global_norm(grads["critic"], args.max_grad_norm),
                }
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), metrics

            return jax.lax.scan(step, (params, opt_state), idx)

        keys = jax.random.split(key, args.epochs)
        (params, opt_state), metrics = jax.lax.scan(epoch, (params, opt_state), keys)
        return params, opt_state, jax.tree_util.tree_map(jp.mean, metrics)

    @jax.jit
    def train_iter(params: dict, opt_state: Any, state: RaceState, p_rsi: Array) -> tuple:
        state, (transitions, episode) = rollout(params, state, p_rsi)
        _, obs_vf = task.observe(state)
        last_value = args.value_scale * critic_forward(params["critic"], obs_vf)
        advantages, returns = gae(transitions, last_value)
        data = {
            "obs_pi": transitions["obs_pi"],
            "obs_vf": transitions["obs_vf"],
            "action": transitions["action"],
            "logp": transitions["logp"],
            "adv": advantages,
            "ret": returns,
        }
        data = {k: v.reshape((batch,) + v.shape[2:]) for k, v in data.items()}
        rng, key = jax.random.split(state.rng)
        params, opt_state, metrics = update(params, opt_state, data, key)
        ended = episode["ended"].sum()
        finished = episode["finished"].sum()
        metrics |= {
            "ended": ended,
            "finished": finished,
            "crashed": episode["crashed"].sum(),
            "lap_sum": jp.sum(episode["time"] * episode["finished"]),
            "passed_sum": jp.sum(episode["passed"] * episode["ended"]),
            "reward": episode["reward"].mean(),
            "value": transitions["value"].mean(),
        }
        return params, opt_state, state.replace(rng=rng), metrics

    return train_iter


def make_eval(task: RaceTask, args: Args) -> Any:
    """Build the jitted evaluation of the deterministic policy from the regular start."""

    @jax.jit
    def evaluate(actor: list, state: RaceState) -> dict:
        state = task.reset(state, jp.ones(task.n, dtype=bool), jp.float32(0.0))

        def body(carry: tuple, _: None) -> tuple:
            state, active, steps, finished, passed = carry
            obs_pi, _ = task.observe(state)
            action = mlp_forward(jp, actor, obs_pi)
            state, _, done, stats = task.step(state, action)
            steps = steps + active
            finished = finished | (active & stats["finished"])
            passed = jp.where(active, stats["passed"], passed)
            return (state, active & ~done, steps, finished, passed), None

        zeros = jp.zeros(task.n, dtype=jp.int32)
        carry = (state, jp.ones(task.n, dtype=bool), zeros, jp.zeros(task.n, dtype=bool), zeros)
        (state, _, steps, finished, passed), _ = jax.lax.scan(
            body, carry, None, length=args.eval_steps
        )
        return {"steps": steps, "finished": finished, "passed": passed, "rng": state.rng}

    return evaluate


def save_policy(path: Path, params: dict, limits: dict[str, float], extra: dict | None = None):
    """Store the actor and the action limits in a plain npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"n_layers": len(params["actor"])}
    for i, (w, b) in enumerate(params["actor"]):
        arrays[f"w{i}"], arrays[f"b{i}"] = np.asarray(w), np.asarray(b)
    arrays |= limits | (extra or {})
    np.savez(path, **arrays)


def load_policy_params(path: str, params: dict, max_tilt: float) -> dict:
    """Load a stored policy into the parameter tree.

    If the tilt limit changes, the roll and pitch outputs are rescaled so that small commands stay
    nearly the same and the policy keeps what it has learned.

    Args:
        path: The stored policy.
        params: The parameter tree to update.
        max_tilt: The tilt limit of the continued training.

    Returns:
        The updated parameter tree.
    """
    ckpt = np.load(path)
    n_layers = int(ckpt["n_layers"])
    actor = [(jp.asarray(ckpt[f"w{i}"]), jp.asarray(ckpt[f"b{i}"])) for i in range(n_layers)]
    w, b = actor[-1]
    factor = float(ckpt["max_tilt"]) / max_tilt
    actor[-1] = (w.at[:, :2].multiply(factor), b.at[:2].multiply(factor))
    params = params | {"actor": actor}
    if "log_std" in ckpt:
        params["log_std"] = jp.asarray(ckpt["log_std"])
    if "critic_n_layers" in ckpt:
        params["critic"] = [
            (jp.asarray(ckpt[f"cw{i}"]), jp.asarray(ckpt[f"cb{i}"]))
            for i in range(int(ckpt["critic_n_layers"]))
        ]
    return params


def critic_arrays(params: dict) -> dict:
    """Arrays of the training state that are only needed to continue training."""
    arrays: dict[str, Any] = {
        "log_std": np.asarray(params["log_std"]),
        "critic_n_layers": len(params["critic"]),
    }
    for i, (w, b) in enumerate(params["critic"]):
        arrays[f"cw{i}"], arrays[f"cb{i}"] = np.asarray(w), np.asarray(b)
    return arrays


def score(success: float, lap: float, min_success: float) -> float:
    """Expected ranking of a policy, lower is better.

    The competition ranks the mean time of the successful runs and needs more than half of the 20
    runs to succeed. Success rates below `min_success` are penalised because 20 runs scatter widely.
    """
    return lap + 8.0 * max(0.0, min_success - success)


def main(config: str = "level2.toml", **kwargs: Any):
    """Train the policy.

    Args:
        config: The race configuration in ``config/``.
        **kwargs: Overrides of the settings in `Args`.
    """
    args = Args(**kwargs)
    print(asdict(args), flush=True)
    race_config = load_config(ROOT / "config" / config)
    race_config.sim.render = False
    params_dyn = load_dynamics_params(race_config.sim.dynamics, race_config.sim.drone)
    thrust_low = 4 * float(params_dyn["thrust_min"])
    thrust_high = 4 * float(params_dyn["thrust_max"])
    limits = {
        "thrust_mid": (thrust_low + thrust_high) / 2.0,
        "thrust_half": (thrust_high - thrust_low) / 2.0,
        "max_tilt": args.max_tilt,
        "mass_nominal": float(params_dyn["mass"]),
    }

    FastContactRaceEnv.drone_inflate = args.drone_inflate
    env = make_env(FastContactRaceEnv, race_config, args, args.num_envs, args.seed)
    task = RaceTask(env, race_config, args, limits)
    eval_env = make_env(VecDroneRaceEnv, race_config, args, args.eval_envs, args.seed + 1000)
    eval_task = RaceTask(eval_env, race_config, args, limits)
    rng = jax.random.PRNGKey(args.seed)
    rng, k_state, k_eval, k_actor, k_critic = jax.random.split(rng, 5)
    state = task.init(k_state)
    eval_state = eval_task.init(k_eval)

    obs_pi, obs_vf = task.observe(state)
    sizes = list(args.hidden)
    params = {
        "actor": init_mlp(k_actor, [obs_pi.shape[-1], *sizes, 3], 0.01),
        "critic": init_mlp(k_critic, [obs_vf.shape[-1], *sizes, 1], 1.0),
        "log_std": jp.full(3, args.log_std_init, dtype=jp.float32),
    }
    if args.init_policy:
        params = load_policy_params(args.init_policy, params, args.max_tilt)
        print(f"Continuing from {args.init_policy}", flush=True)

    batch = args.num_envs * args.num_steps
    iterations = args.total_steps // batch
    n_updates = iterations * args.epochs * args.minibatches
    schedule = optax.linear_schedule(args.lr, args.lr_final, n_updates)
    optimizer = optax.adam(schedule, eps=1e-5)
    opt_state = optimizer.init(params)
    train_iter = make_train_iter(task, args, optimizer)
    evaluate = make_eval(eval_task, args)

    run_dir = CHECKPOINT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best = np.inf
    start = time.perf_counter()
    window: dict[str, float] = {}
    for iteration in range(1, iterations + 1):
        frac = (iteration - 1) / max(iterations - 1, 1)
        p_rsi = jp.float32(args.rsi_start + frac * (args.rsi_end - args.rsi_start))
        params, opt_state, state, metrics = train_iter(params, opt_state, state, p_rsi)
        metrics = {k: float(v) for k, v in metrics.items()}
        for k in ("ended", "finished", "crashed", "lap_sum", "passed_sum"):
            window[k] = window.get(k, 0.0) + metrics[k]

        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.perf_counter() - start
            ended = max(window["ended"], 1.0)
            lap = window["lap_sum"] / max(window["finished"], 1.0)
            print(
                f"{iteration:5d}/{iterations} {iteration * batch / 1e6:6.1f}M "
                f"{iteration * batch / elapsed:7.0f} sps | rew {metrics['reward']:6.3f} "
                f"val {metrics['value']:6.2f} vloss {metrics['value_loss']:.2f} "
                f"kl {metrics['kl']:.4f} "
                f"std {np.exp(np.asarray(params['log_std'])).round(3)} | "
                f"start episodes {window['ended']:5.0f} finish {window['finished'] / ended:5.1%} "
                f"crash {window['crashed'] / ended:5.1%} gates {window['passed_sum'] / ended:4.2f} "
                f"lap {lap:5.2f}",
                flush=True,
            )
            window = {}

        if iteration % args.eval_every == 0 or iteration == iterations:
            result = evaluate(params["actor"], eval_state)
            eval_state = eval_state.replace(rng=result["rng"])
            finished = np.asarray(result["finished"])
            success = float(finished.mean())
            laps = (np.asarray(result["steps"])[finished] - 1) * eval_task.dt
            lap = float(laps.mean()) if len(laps) else np.inf
            value = score(success, lap, args.min_success)
            print(
                f"  EVAL {iteration * batch / 1e6:6.1f}M: success {success:6.1%} "
                f"lap {lap:5.3f} (min {laps.min() if len(laps) else np.inf:5.2f}, "
                f"std {laps.std() if len(laps) else 0.0:4.2f}) "
                f"gates {np.asarray(result['passed']).mean():4.2f} score {value:6.3f}",
                flush=True,
            )
            extra = critic_arrays(params)
            save_policy(run_dir / "latest.npz", params, limits, extra)
            if value < best:
                best = value
                save_policy(run_dir / "best.npz", params, limits, extra)
                print(f"  new best {best:.3f}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
