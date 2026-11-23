"""The drone racing problem in the form the generic MPPI engine (``mppi_engine.py``) expects.

This module provides the three ingredients of the engine:

* ``dynamics(x, u, params)``: The identified ``so_rpy`` drone model of Crazyflow, plus the
  bookkeeping a race needs along a rollout (next gate to pass, crashed or not, input change).
* ``running_cost(x_next, u, params)``: Progress through the gate sequence, collision penalties,
  soft safety margins and input regularization.
* ``params``: A ``RacingParams`` pytree with the model parameters, the current (sensed) track
  layout and the disturbance estimate. It is rebuilt from the observation in every control step.

Both functions also take a ``RacingConfig`` with the cost weights. It is fixed for the lifetime of
the controller, so bind it with ``functools.partial`` before handing the functions to the engine.

Rollout state ``x`` (a ``RolloutState`` pytree):
    drone: [pos(3), rpy(3), vel(3), rpy_rates(3)], the state of the so_rpy model.
    u_last: The input of the previous step, needed for the smoothness cost.
    du: Input change of the step that led to this state.
    gate: Index into the gate sequence of the next gate to pass (= number of gates passed).
    crashed: Whether the rollout has hit something so far.
    new_crash: Whether the rollout hit something in the step that led to this state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Collision geometry from lsy_drone_racing/envs/assets/{gate,obstacle}.xml, expressed in the gate
# frame. Gates are passed along their local x-axis, the frame lies in the local y-z plane.
GATE_INNER = 0.2  # Half width of the gate opening [m]
GATE_OUTER = 0.36  # Half width of the outer frame [m]
GATE_HALF_THICKNESS = 0.01  # [m]
STAND_HALF_WIDTH = 0.05  # The stand below each gate [m]
OBSTACLE_RADIUS = 0.015  # Obstacles are thin vertical poles [m]


class RacingConfig(NamedTuple):
    """Model settings and cost weights of the racing problem."""

    dt: float = 0.02  # Rollout step length [s], set to 1 / env freq
    max_tilt: float = 0.6  # Roll and pitch command limit [rad]
    # Progress cost: remaining race distance, measured through an entry point in front of each gate
    entry_dist: float = 0.4  # Distance of the entry point in front of the gate [m]
    funnel_radius_entry: float = 0.25  # Radius of the approach funnel at the entry point [m]
    funnel_radius_gate: float = 0.1  # Radius of the approach funnel at the gate [m]
    w_progress: float = 1.0
    # Collisions
    drone_radius: float = 0.07  # Used for collision checks [m]
    crash_cost: float = 1000.0  # Paid once per rollout at the first collision
    # Soft safety margins
    obstacle_safe_dist: float = 0.25  # Clearance to the poles [m]
    w_obstacle: float = 100.0
    gate_safe_dist: float = 0.15  # Clearance to the gate frames [m]
    w_gate: float = 100.0
    z_min: float = 0.15  # Minimum altitude [m]
    w_ground: float = 100.0
    v_max: float = 2.0  # Speed limit [m/s]
    w_speed: float = 10.0
    # Input regularization
    w_tilt: float = 0.5
    w_smooth: tuple[float, float, float, float] = (5.0, 5.0, 0.0, 20.0)  # On input changes


class RolloutState(NamedTuple):
    """State carried through a rollout. See the module docstring for the fields."""

    drone: jax.Array
    u_last: jax.Array
    du: jax.Array
    gate: jax.Array
    crashed: jax.Array
    new_crash: jax.Array


class RacingParams(NamedTuple):
    """Everything the rollout needs that may change from one control step to the next."""

    model: dict[str, jax.Array]  # so_rpy model parameters, see load_model_params
    track: dict[str, jax.Array]  # Track layout, see build_track
    acc_bias: jax.Array  # Estimated unmodelled acceleration [m/s^2]


# region Model


def load_model_params(drone: str) -> dict[str, jax.Array]:
    """Identified so_rpy parameters of the given drone model, e.g. "cf21B_500"."""
    p = load_params(Dynamics.so_rpy, drone)
    keys = ("mass", "gravity_vec", "acc_coef", "cmd_f_coef")
    keys += ("rpy_coef", "rpy_rates_coef", "cmd_rpy_coef")
    model = {k: jnp.asarray(p[k], dtype=jnp.float32) for k in keys}
    model["thrust_min"] = jnp.float32(4 * p["thrust_min"])  # Collective thrust limits [N]
    model["thrust_max"] = jnp.float32(4 * p["thrust_max"])
    return model


def hover_thrust(model: dict[str, jax.Array]) -> float:
    """Collective thrust command that keeps the drone hovering."""
    weight = model["mass"] * -model["gravity_vec"][2]
    return float((weight - model["acc_coef"]) / model["cmd_f_coef"])


def input_bounds(model: dict[str, jax.Array], cfg: RacingConfig) -> tuple[tuple, tuple]:
    """Lower and upper bounds of the attitude command [roll, pitch, yaw, thrust]."""
    u_min = (-cfg.max_tilt, -cfg.max_tilt, -np.pi, float(model["thrust_min"]))
    u_max = (cfg.max_tilt, cfg.max_tilt, np.pi, float(model["thrust_max"]))
    return u_min, u_max


def obs_to_drone_state(obs: dict[str, NDArray[np.floating]]) -> jax.Array:
    """Convert the observation into the model state [pos, rpy, vel, rpy_rates]."""
    rpy = R.from_quat(obs["quat"]).as_euler("xyz")
    rpy_rates = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
    x = np.concatenate([obs["pos"], rpy, obs["vel"], rpy_rates])
    return jnp.asarray(x, dtype=jnp.float32)


def body_z_axis(rpy: jax.Array) -> jax.Array:
    """Third column of R = Rz(yaw) Ry(pitch) Rx(roll), i.e. scipy's extrinsic "xyz" convention."""
    cr, sr = jnp.cos(rpy[0]), jnp.sin(rpy[0])
    cp, sp = jnp.cos(rpy[1]), jnp.sin(rpy[1])
    cy, sy = jnp.cos(rpy[2]), jnp.sin(rpy[2])
    return jnp.stack([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])


def so_rpy_step(
    x: jax.Array, u: jax.Array, model: dict, acc_bias: jax.Array, dt: float
) -> jax.Array:
    """One semi-implicit Euler step of the so_rpy model (``crazyflow.dynamics.so_rpy``).

    Args:
        x: State [pos(3), rpy(3), vel(3), rpy_rates(3)].
        u: Command [roll, pitch, yaw, thrust], the env's attitude action format.
        model: Model parameters from load_model_params.
        acc_bias: Estimated unmodelled acceleration [m/s^2].
        dt: Step length [s].
    """
    pos, rpy, vel, rpy_rates = x[0:3], x[3:6], x[6:9], x[9:12]
    rpy_acc = (
        model["rpy_coef"] * rpy
        + model["rpy_rates_coef"] * rpy_rates
        + model["cmd_rpy_coef"] * u[:3]
    )
    rpy_rates = rpy_rates + dt * rpy_acc
    rpy = rpy + dt * rpy_rates
    thrust = model["acc_coef"] + model["cmd_f_coef"] * u[3]
    acc = thrust / model["mass"] * body_z_axis(rpy) + model["gravity_vec"] + acc_bias
    vel = vel + dt * acc
    pos = pos + dt * vel
    return jnp.concatenate([pos, rpy, vel, rpy_rates])


# region Track


def build_track(
    obs: dict[str, NDArray[np.floating]],
    pos_low: NDArray[np.floating],
    pos_high: NDArray[np.floating],
    cfg: RacingConfig,
) -> dict[str, jax.Array]:
    """Track layout from the current observation. Computed with numpy, outside of jit.

    Gates and obstacles are needed in two forms: all gates for collision checks, and the gates in
    the order of the race ("passes", one entry per gate in the sequence) for progress and gate
    passing. Shapes and dtypes are always the same, so the jitted MPPI step never recompiles.
    """
    gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
    gates_rot = R.from_quat(obs["gates_quat"]).as_matrix().astype(np.float32)
    seq = np.asarray(obs["gate_sequence"], dtype=np.int32)
    direction = np.asarray(obs["gate_sequence_direction"], dtype=np.float32)

    pass_pos, pass_rot = gates_pos[seq], gates_rot[seq]
    pass_normal = pass_rot[:, :, 0] * direction[:, None]  # Points in the passing direction
    pass_entry = pass_pos - cfg.entry_dist * pass_normal

    # remaining[k]: path length from the center of pass k to the finish. The path leads from each
    # gate center through the entry point of the next gate to its center.
    segments = np.linalg.norm(pass_entry[1:] - pass_pos[:-1], axis=-1) + cfg.entry_dist
    remaining = np.append(np.cumsum(segments[::-1])[::-1], 0.0)

    track = {
        "gates_pos": gates_pos,
        "gates_rot": gates_rot,
        "pass_pos": pass_pos,
        "pass_rot": pass_rot,
        "pass_dir": direction,
        "pass_entry": pass_entry,
        "remaining": remaining,
        "obstacles_pos": np.asarray(obs["obstacles_pos"], dtype=np.float32),
        "pos_low": np.asarray(pos_low),
        "pos_high": np.asarray(pos_high),
    }
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in track.items()}


def n_passes(track: dict[str, jax.Array]) -> int:
    """Number of gate passes in the race (static, known at trace time)."""
    return track["pass_pos"].shape[0]


def gate_local(pos: jax.Array, k: jax.Array, track: dict) -> tuple[jax.Array, jax.Array]:
    """Position in the frame of pass k: (signed distance along the passing direction, y-z)."""
    k = jnp.minimum(k, n_passes(track) - 1)
    local = track["pass_rot"][k].T @ (pos - track["pass_pos"][k])
    return local[0] * track["pass_dir"][k], local[1:]


def gates_local(pos: jax.Array, track: dict) -> jax.Array:
    """Position in the frame of every gate, shape (n_gates, 3)."""
    return jnp.einsum("gji,gj->gi", track["gates_rot"], pos - track["gates_pos"])  # R^T (p - c)


# region Cost terms


def distance_to_go(pos: jax.Array, k: jax.Array, track: dict, cfg: RacingConfig) -> jax.Array:
    """Remaining race distance when pass k is the next target.

    Inside a funnel in front of the gate we measure the distance to the gate center. Outside of
    it, we measure the distance to the entry point in front of the gate plus the distance from the
    entry point to the gate. This keeps the drone from approaching gates from the side, and the
    cost is (almost) continuous when a gate is passed and the target switches to the next one.
    """
    kc = jnp.minimum(k, n_passes(track) - 1)
    center, entry = track["pass_pos"][kc], track["pass_entry"][kc]
    along, yz = gate_local(pos, k, track)
    lateral = jnp.linalg.norm(yz)

    frac = jnp.clip(-along / cfg.entry_dist, 0.0, 1.0)  # 0 at the gate, 1 at the entry point
    radius = cfg.funnel_radius_gate + (cfg.funnel_radius_entry - cfg.funnel_radius_gate) * frac
    in_funnel = (along > -cfg.entry_dist) & (along <= 0.0) & (lateral < radius)
    to_gate = jnp.where(
        in_funnel, jnp.linalg.norm(pos - center), jnp.linalg.norm(pos - entry) + cfg.entry_dist
    )
    return jnp.where(k >= n_passes(track), 0.0, to_gate + track["remaining"][kc])


def gate_passed(pos_prev: jax.Array, pos: jax.Array, k: jax.Array, track: dict) -> jax.Array:
    """Whether the segment pos_prev -> pos passes through the opening of pass k correctly."""
    x0, yz0 = gate_local(pos_prev, k, track)
    x1, yz1 = gate_local(pos, k, track)
    crossed = (x0 < 0.0) & (x1 >= 0.0)
    alpha = -x0 / jnp.where(crossed, x1 - x0, 1.0)  # Interpolate the plane intersection
    yz = yz0 + alpha * (yz1 - yz0)
    return crossed & jnp.all(jnp.abs(yz) < GATE_INNER) & (k < n_passes(track))


def collision(pos: jax.Array, track: dict, cfg: RacingConfig) -> jax.Array:
    """Collision with any gate frame, gate stand, obstacle pole, or the safety limits."""
    r = cfg.drone_radius
    local = gates_local(pos, track)
    ax, ay, z = jnp.abs(local[:, 0]), jnp.abs(local[:, 1]), local[:, 2]
    square_dist = jnp.maximum(ay, jnp.abs(z))  # Gates are square, so use the inf-norm
    frame = (
        (ax < GATE_HALF_THICKNESS + r)
        & (square_dist > GATE_INNER - r)
        & (square_dist < GATE_OUTER + r)
    )
    stand = (ax < STAND_HALF_WIDTH + r) & (ay < STAND_HALF_WIDTH + r) & (z < -GATE_OUTER)

    obstacles = track["obstacles_pos"]
    d_pole = jnp.linalg.norm(pos[:2] - obstacles[:, :2], axis=-1)
    pole = (d_pole < OBSTACLE_RADIUS + r) & (pos[2] < obstacles[:, 2] + r)

    out_of_bounds = jnp.any(pos < track["pos_low"]) | jnp.any(pos > track["pos_high"])
    return jnp.any(frame | stand) | jnp.any(pole) | out_of_bounds


def safety_cost(pos: jax.Array, vel: jax.Array, track: dict, cfg: RacingConfig) -> jax.Array:
    """Quadratic penalties for coming too close to poles, gate frames, the ground, or too fast."""
    obstacles = track["obstacles_pos"]
    d_pole = jnp.linalg.norm(pos[:2] - obstacles[:, :2], axis=-1)
    below_top = pos[2] < obstacles[:, 2] + cfg.obstacle_safe_dist
    pole = jnp.sum(jnp.maximum(cfg.obstacle_safe_dist - d_pole, 0.0) ** 2 * below_top)

    # Distance to the square frame of each gate: in the gate plane, then combined with the offset
    local = gates_local(pos, track)
    square_dist = jnp.maximum(jnp.abs(local[:, 1]), jnp.abs(local[:, 2]))
    in_plane = jnp.maximum(jnp.maximum(GATE_INNER - square_dist, square_dist - GATE_OUTER), 0.0)
    d_frame = jnp.sqrt(local[:, 0] ** 2 + in_plane**2)
    frame = jnp.sum(jnp.maximum(cfg.gate_safe_dist - d_frame, 0.0) ** 2)

    ground = jnp.maximum(cfg.z_min - pos[2], 0.0) ** 2
    speed = jnp.maximum(jnp.linalg.norm(vel) - cfg.v_max, 0.0) ** 2
    return cfg.w_obstacle * pole + cfg.w_gate * frame + cfg.w_ground * ground + cfg.w_speed * speed


# region MPPI interface


def initial_state(drone: jax.Array, u_last: jax.Array, gates_passed: int) -> RolloutState:
    """Rollout start state from the measured drone state and the last applied input."""
    return RolloutState(
        drone=drone,
        u_last=jnp.asarray(u_last, dtype=jnp.float32),
        du=jnp.zeros(4, dtype=jnp.float32),
        gate=jnp.int32(gates_passed),
        crashed=jnp.array(False),
        new_crash=jnp.array(False),
    )


def dynamics(
    x: RolloutState, u: jax.Array, params: RacingParams, cfg: RacingConfig
) -> RolloutState:
    """Advance the drone by one step and record the race events of that step."""
    drone = so_rpy_step(x.drone, u, params.model, params.acc_bias, cfg.dt)
    pos_prev, pos = x.drone[:3], drone[:3]
    passed = gate_passed(pos_prev, pos, x.gate, params.track) & ~x.crashed
    hit = collision(pos, params.track, cfg)
    return RolloutState(
        drone=drone,
        u_last=u,
        du=u - x.u_last,
        gate=x.gate + passed,
        crashed=x.crashed | hit,
        new_crash=hit & ~x.crashed,
    )


def running_cost(
    x: RolloutState, u: jax.Array, params: RacingParams, cfg: RacingConfig
) -> jax.Array:
    """Cost of the step that applied u and led to x."""
    pos, vel = x.drone[0:3], x.drone[6:9]
    cost = cfg.w_progress * distance_to_go(pos, x.gate, params.track, cfg)
    cost += cfg.crash_cost * x.new_crash
    cost += safety_cost(pos, vel, params.track, cfg)
    cost += cfg.w_tilt * (u[0] ** 2 + u[1] ** 2)
    cost += jnp.sum(jnp.asarray(cfg.w_smooth) * x.du**2)
    return cost
