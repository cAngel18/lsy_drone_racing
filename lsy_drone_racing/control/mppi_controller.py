"""Model Predictive Path Integral (MPPI) controller for drone racing.

This is a basic, self-contained MPPI implementation that races through randomized tracks (level 3)
using the collective thrust and attitude interface. It is intended as a starting point to understand
the algorithm and to experiment with it.

Algorithm (one iteration per control step, Williams et al. 2017):

1. **Sample** ``K`` perturbed control sequences ``u_k = U + eps_k`` around the nominal sequence
   ``U`` (shape ``H x 3``: roll, pitch, collective thrust). The noise ``eps`` is temporally
   correlated (first-order low-pass filtered), which yields smoother and more useful samples than
   white noise.
2. **Roll out** every sequence with a model of the drone (the identified ``so_rpy`` model from
   crazyflow, which the simulator's attitude interface is fitted to). All rollouts run in parallel
   with ``jax.vmap`` and ``jax.lax.scan``.
3. **Evaluate** each rollout with a cost function: progress along a path through the gates, a
   corridor around the path, and non-smooth penalties for collisions with gates, obstacles, and the
   arena bounds. Sampling-based MPC handles such discontinuous costs naturally.
4. **Weight** the samples with ``w_k = exp(-(S_k - min S) / lambda)`` (softmax with temperature
   ``lambda``) and update the nominal sequence with the weighted noise ``U <- U + sum_k w_k eps_k``.
5. **Apply** the first control ``U[0]``, then **shift** ``U`` by one step to warm-start the next
   iteration.

Because the track is only known at runtime (and gate/obstacle positions change once they come into
sensor range), a lightweight planner (grid search between the gates + cubic spline) builds a path
through all remaining gates and re-plans whenever the observed track changes. MPPI maximizes the
progress along this path inside a corridor and below the planner's curvature-limited speed profile,
so it chooses its own racing line (contouring-style). Its collision costs act as a safety layer.
Setting ``progress_cost=False`` switches to tracking a reference point that moves along the path.

A small disturbance observer estimates a constant acceleration offset (e.g. from the randomized
mass) by comparing predicted and measured velocities, and feeds it back into the rollouts.

Visualization (``render=True`` in the config or ``--render True``):

* green line: planned path through the gates,
* small blue spheres: the part of the path MPPI makes progress along (or the tracked reference
  points with ``progress_cost=False``),
* thin lines: a subset of the sampled rollouts, colored by their MPPI weight (green = high weight
  / low cost, red = low weight / high cost, purple = predicted collision),
* thick yellow line: predicted trajectory of the updated (weighted mean) control sequence,
* magenta sphere and line: current target gate and its required passing direction,
* orange rings: obstacle safety radius used by the cost function, at the drone's height.

Run with:

    python scripts/sim.py --config level3.toml --controller mppi_controller.py --render True

Note:
    This controller requires ``control_mode = "attitude"`` in the ``[env]`` section of the config.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Gate and obstacle geometry from `lsy_drone_racing/envs/assets/{gate,obstacle}.xml`.
GATE_INNER_HALF = 0.20  # Half width of the gate opening [m]
GATE_OUTER_HALF = 0.36  # Half width of the outer frame [m]
GATE_FRAME_HALF_DEPTH = 0.01  # Half thickness of the frame along the gate normal [m]
GATE_STAND_HALF = 0.05  # Half width of the gate stand below the frame [m]
GATE_STAND_LENGTH = 1.0  # Length of the stand below the frame [m]


# region Settings


@dataclass(frozen=True)
class MPPIConfig:
    """Hyperparameters of the MPPI controller and the reference planner.

    The dataclass is frozen (hashable) so that it can be passed as a static argument to the jitted
    MPPI step. Changing a value triggers a single recompilation.
    """

    # --- Sampling ---
    n_samples: int = 1024  # Number of sampled control sequences K
    horizon: int = 40  # Prediction horizon H in control steps (H * dt seconds)
    dt: float = 0.02  # Control period [s], set from the environment frequency
    n_substeps: int = 2  # Integration substeps per control step
    temperature: float = 20.0  # MPPI temperature lambda. Low: greedy, high: average of all samples
    noise_sigma: tuple[float, float, float] = (
        0.2,
        0.2,
        0.1,
    )  # Std of roll, pitch [rad], thrust [N]
    noise_beta: float = 0.8  # Temporal correlation of the noise (0 = white noise, ->1 = constant)
    gamma: float = 0.05  # Weight of the information-theoretic control cost lambda * U^T S^-1 eps
    max_tilt: float = 0.9  # Roll/pitch command limit [rad]

    # --- Cost formulation ---
    # True: maximize progress along the path inside a corridor, MPPI picks its own racing line and
    # speed (contouring-style). False: track a reference point that moves along the path.
    progress_cost: bool = True
    w_progress: float = 300.0  # Reward per meter of progress along the path within the horizon
    w_contour: float = 2000.0  # Leaving the corridor around the path
    w_center: float = 5.0  # Mild pull towards the center of the corridor
    w_speed: float = 30.0  # Exceeding the speed profile of the planner (looks beyond the horizon)
    corridor_width: float = 0.25  # Corridor half width away from gates and obstacles [m]
    n_window: int = 96  # Number of path points MPPI sees for the progress cost

    # --- Cost weights ---
    w_pos: float = 80.0  # Tracking of the reference position
    w_vel: float = 2.0  # Tracking of the reference velocity
    w_term_pos: float = 300.0  # Terminal position error
    w_term_vel: float = 10.0  # Terminal velocity error
    w_att: float = 1.0  # Roll and pitch angles (keeps the drone calm)
    w_du: tuple[float, float, float] = (5.0, 5.0, 50.0)  # Control rate (smoothness)
    w_obs: float = 3000.0  # Soft obstacle clearance cost
    w_gate: float = 600.0  # Soft cost to stay away from gate frames
    w_gate_center: float = 200.0  # Attraction to the center line of the target gate
    w_bounds: float = 500.0  # Soft arena bounds cost
    w_crash: float = 5000.0  # Cost per predicted step in collision (accumulates after a crash)

    # --- Geometry and safety margins ---
    r_drone: float = 0.08  # Collision radius of the drone [m]
    r_obs_hit: float = 0.15  # Obstacle distance considered a collision [m]
    r_obs_safe: float = 0.22  # Obstacle distance below which the soft cost starts [m]
    gate_margin: float = 0.02  # Additional soft margin around gate frames [m]
    z_min: float = 0.1  # Soft lower bound [m]

    # --- Reference path and speed profile ---
    v_max: float = 6.0  # Maximum reference speed [m/s]
    v_min: float = 0.4  # Minimum reference speed, so that the reference never gets stuck [m/s]
    a_max: float = 12.0  # Acceleration limit of the reference [m/s^2]
    a_lat: float = 10.0  # Lateral acceleration limit used for the speed profile [m/s^2]
    d_pre: float = 0.35  # Distance of the approach waypoint in front of each gate [m]
    d_post: float = 0.35  # Distance of the exit waypoint behind each gate [m]
    gate_axis_clearance: float = 0.3  # Preferred obstacle clearance of gate approach/exit segments
    gate_axis_min_clearance: float = 0.16  # Approach/exit segments are shortened below this [m]
    obs_clearance: float = 0.25  # Planner: obstacles closer than this block the path [m]
    gate_frame_clearance: float = 0.15  # Planner: gate frames closer than this block the path [m]
    grid_res: float = 0.05  # Resolution of the planning grid [m]
    grid_r_soft: float = 0.5  # Distance below which the planner prefers to stay away [m]
    grid_w_clear: float = 4.0  # Weight of the clearance cost in the grid search
    v_unseen_gate: float = 2.5  # Speed limit in front of gates that have not been observed [m/s]
    unseen_gate_window: float = 0.5  # Length of the speed limited section before such gates [m]

    # --- Disturbance observer ---
    dist_alpha: tuple[float, float, float] = (0.02, 0.02, 0.05)  # Filter gains (x, y, z)
    dist_limit: tuple[float, float, float] = (1.5, 1.5, 3.0)  # Saturation [m/s^2]

    # --- Visualization ---
    n_vis: int = 24  # Number of rollouts to draw
    vis_stride: int = 2  # Draw every n-th point of each rollout


class ModelParams(NamedTuple):
    """Parameters of the identified so_rpy model (a JAX pytree).

    The attitude dynamics of the so_rpy model are linear and decoupled per axis,
    ``ddrpy = c1 * rpy + c2 * drpy + c3 * rpy_cmd``. We discretize them exactly (zero-order hold)
    with the matrix exponential, which is much more accurate than explicit Euler for these stiff
    dynamics at no additional cost.
    """

    mass: jax.Array  # ()
    gravity: jax.Array  # (3,)
    acc_coef: jax.Array  # ()
    cmd_f_coef: jax.Array  # ()
    att_ad: jax.Array  # (3, 2, 2) discrete attitude dynamics [angle, rate] per axis
    att_bd: jax.Array  # (3, 2) discrete attitude input matrix per axis
    dt_sub: jax.Array  # () duration of one integration substep [s]


def build_model_params(p: dict, h: float) -> ModelParams:
    """Create the model parameters for an integration substep of ``h`` seconds.

    Args:
        p: so_rpy parameters as returned by ``crazyflow.dynamics.load_params``.
        h: Duration of one integration substep [s].
    """
    ad, bd = np.zeros((3, 2, 2)), np.zeros((3, 2))
    for i in range(3):
        # Continuous system [angle, rate]' = A [angle, rate] + B cmd, augmented for ZOH
        m = np.zeros((3, 3))
        m[0, 1] = 1.0
        m[1, 0], m[1, 1], m[1, 2] = p["rpy_coef"][i], p["rpy_rates_coef"][i], p["cmd_rpy_coef"][i]
        md = expm(m * h)
        ad[i], bd[i] = md[:2, :2], md[:2, 2]
    return ModelParams(
        mass=jnp.asarray(p["mass"], dtype=jnp.float32),
        gravity=jnp.asarray(p["gravity_vec"], dtype=jnp.float32),
        acc_coef=jnp.asarray(p["acc_coef"], dtype=jnp.float32),
        cmd_f_coef=jnp.asarray(p["cmd_f_coef"], dtype=jnp.float32),
        att_ad=jnp.asarray(ad, dtype=jnp.float32),
        att_bd=jnp.asarray(bd, dtype=jnp.float32),
        dt_sub=jnp.asarray(h, dtype=jnp.float32),
    )


class TrackArrays(NamedTuple):
    """Track information used inside the cost function (a JAX pytree)."""

    gates_pos: jax.Array  # (G, 3)
    gates_rot: jax.Array  # (G, 3, 3) rotation matrices gate -> world
    obstacles_pos: jax.Array  # (O, 3), z is the top of the pole
    target_gate: jax.Array  # () index of the gate that has to be passed next
    pos_low: jax.Array  # (3,) soft arena bounds
    pos_high: jax.Array  # (3,)


class PathWindow(NamedTuple):
    """The part of the planned path ahead of the drone, used by the progress cost (a JAX pytree).

    The points are equally spaced in arc length, starting slightly behind the drone.
    """

    pos: jax.Array  # (M, 3) path points
    tangent: jax.Array  # (M, 3) unit tangents
    s: jax.Array  # (M,) arc length relative to the drone's current progress
    width: jax.Array  # (M,) corridor half width
    v_max: jax.Array  # (M,) speed limit from the planner's speed profile
    idx0: jax.Array  # () index of the point closest to the drone


class MPPIInfo(NamedTuple):
    """Diagnostics returned by the MPPI step (mostly for visualization)."""

    vis_rollouts: jax.Array  # (n_vis, H, 3) positions of a subset of sampled rollouts
    vis_cost: jax.Array  # (n_vis,) cost of these rollouts relative to the best sample / lambda
    vis_crashed: jax.Array  # (n_vis,) whether these rollouts are predicted to collide
    opt_traj: jax.Array  # (H, 3) predicted positions of the updated control sequence
    min_cost: jax.Array  # () cost of the best sample
    ess: jax.Array  # () effective sample size 1 / sum(w^2)
    crash_ratio: jax.Array  # () fraction of samples that are predicted to collide


# region MPPI core (JAX)


def dynamics_step(
    x: jax.Array, u: jax.Array, params: ModelParams, dist_acc: jax.Array, n_sub: int
) -> jax.Array:
    """Integrate the so_rpy model for one control step.

    The attitude is propagated exactly (see
    [ModelParams][lsy_drone_racing.control.mppi_controller.ModelParams]), the translational
    dynamics with the trapezoidal rule.

    Args:
        x: State [pos(3), vel(3), rpy(3), rpy_rates(3)].
        u: Control [roll_cmd, pitch_cmd, collective_thrust]. The yaw command is fixed to 0.
        params: Model parameters, discretized for a substep of ``dt / n_sub``.
        dist_acc: Estimated constant disturbance acceleration in the world frame (3,).
        n_sub: Number of integration substeps per control step.

    Returns:
        The next state.
    """
    rpy_cmd = jnp.array([u[0], u[1], 0.0])
    thrust = (params.acc_coef + params.cmd_f_coef * u[2]) / params.mass
    h = params.dt_sub

    def acceleration(rpy: jax.Array) -> jax.Array:
        cr, sr = jnp.cos(rpy[0]), jnp.sin(rpy[0])
        cp, sp = jnp.cos(rpy[1]), jnp.sin(rpy[1])
        cy, sy = jnp.cos(rpy[2]), jnp.sin(rpy[2])
        # Body z-axis in world frame for extrinsic xyz Euler angles (R = Rz @ Ry @ Rx)
        z_axis = jnp.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])
        return z_axis * thrust + params.gravity + dist_acc

    pos, vel = x[0:3], x[3:6]
    att = jnp.stack([x[6:9], x[9:12]], axis=-1)  # (3, 2): [angle, rate] per axis
    acc = acceleration(att[:, 0])
    for _ in range(n_sub):
        att = jnp.einsum("aij,aj->ai", params.att_ad, att) + params.att_bd * rpy_cmd[:, None]
        acc_next = acceleration(att[:, 0])
        vel_next = vel + 0.5 * h * (acc + acc_next)
        pos = pos + 0.5 * h * (vel + vel_next)
        vel, acc = vel_next, acc_next
    return jnp.concatenate([pos, vel, att[:, 0], att[:, 1]])


def rollout(
    x0: jax.Array, us: jax.Array, params: ModelParams, dist_acc: jax.Array, n_sub: int
) -> jax.Array:
    """Simulate a control sequence ``us`` (H, 3) from ``x0``. Returns the states (H, 12)."""

    def step(x: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_next = dynamics_step(x, u, params, dist_acc, n_sub)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, us)
    return xs


def sample_noise(key: jax.Array, cfg: MPPIConfig) -> jax.Array:
    """Sample temporally correlated Gaussian noise of shape (K, H, 3).

    Each sequence is a first-order autoregressive process
    ``e_t = beta * e_{t-1} + sqrt(1 - beta^2) * w_t`` with white noise ``w_t``. This is
    equivalent to low-pass filtering the white noise while keeping the stationary standard deviation
    at ``noise_sigma``.
    """
    sigma = jnp.asarray(cfg.noise_sigma)
    white = jax.random.normal(key, (cfg.horizon, cfg.n_samples, 3)) * sigma
    beta = cfg.noise_beta

    def low_pass(prev: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
        e = beta * prev + jnp.sqrt(1.0 - beta**2) * w
        return e, e

    _, eps = jax.lax.scan(low_pass, white[0], white[1:])
    eps = jnp.concatenate([white[:1], eps], axis=0)  # (H, K, 3)
    return jnp.swapaxes(eps, 0, 1)


def trajectory_cost(
    xs: jax.Array,
    us: jax.Array,
    u_prev: jax.Array,
    ref_pos: jax.Array,
    ref_vel: jax.Array,
    window: PathWindow,
    track: TrackArrays,
    cfg: MPPIConfig,
) -> tuple[jax.Array, jax.Array]:
    """Compute the cost of a single rollout.

    Args:
        xs: Predicted states (H, 12).
        us: Applied controls (H, 3).
        u_prev: Control applied in the previous step (3,), used for the smoothness cost.
        ref_pos: Reference positions (H, 3), only used if ``cfg.progress_cost`` is False.
        ref_vel: Reference velocities (H, 3), only used if ``cfg.progress_cost`` is False.
        window: Path ahead of the drone, only used if ``cfg.progress_cost`` is True.
        track: Gates, obstacles and bounds.
        cfg: MPPI settings.

    Returns:
        The total cost and a flag whether the rollout collides.
    """
    pos, vel, rpy = xs[:, 0:3], xs[:, 3:6], xs[:, 6:9]
    r = cfg.r_drone

    if cfg.progress_cost:
        # Reward progress along the path, stay inside the corridor, respect the speed profile
        s, err, idx = project_on_window(pos, window)
        c_track = cfg.w_contour * jnp.maximum(err - window.width[idx], 0.0) ** 2
        c_track += cfg.w_center * err**2
        c_track += (
            cfg.w_speed * jnp.maximum(jnp.linalg.norm(vel, axis=-1) - window.v_max[idx], 0) ** 2
        )
        terminal = -cfg.w_progress * s[-1]
    else:
        # Tracking of the moving reference point on the path
        c_track = cfg.w_pos * jnp.sum((pos - ref_pos) ** 2, axis=-1)
        c_track += cfg.w_vel * jnp.sum((vel - ref_vel) ** 2, axis=-1)
        terminal = cfg.w_term_pos * jnp.sum((pos[-1] - ref_pos[-1]) ** 2)
        terminal += cfg.w_term_vel * jnp.sum((vel[-1] - ref_vel[-1]) ** 2)

    # Regularization: small attitude and smooth controls
    c_ctrl = cfg.w_att * jnp.sum(rpy[:, :2] ** 2, axis=-1)
    du = jnp.diff(jnp.concatenate([u_prev[None], us], axis=0), axis=0)
    c_ctrl += jnp.sum(jnp.asarray(cfg.w_du) * du**2, axis=-1)

    # Obstacles are vertical poles from the ground to the top position
    d_obs = jnp.linalg.norm(pos[:, None, :2] - track.obstacles_pos[None, :, :2], axis=-1)  # (H, O)
    below_top = pos[:, None, 2] < track.obstacles_pos[None, :, 2] + r
    c_obs = cfg.w_obs * jnp.sum(below_top * jnp.maximum(cfg.r_obs_safe - d_obs, 0.0) ** 2, axis=-1)
    hit_obs = jnp.any(below_top & (d_obs < cfg.r_obs_hit), axis=-1)

    # Gates: express positions in each gate frame. x is the gate normal, y lateral, z up
    rel = pos[:, None, :] - track.gates_pos[None]  # (H, G, 3)
    q = jnp.einsum("gji,hgj->hgi", track.gates_rot, rel)  # R^T (p - g)
    ax, ay, az = jnp.abs(q[..., 0]), jnp.abs(q[..., 1]), q[..., 2]
    abs_az = jnp.abs(az)

    def frame_region(m: float) -> jax.Array:
        """Region around the gate frame, inflated by drone radius + m."""
        in_slab = ax < GATE_FRAME_HALF_DEPTH + r + m
        in_outer = (ay < GATE_OUTER_HALF + r + m) & (abs_az < GATE_OUTER_HALF + r + m)
        in_inner = (ay < GATE_INNER_HALF - r - m) & (abs_az < GATE_INNER_HALF - r - m)
        in_stand = (
            (az < -GATE_OUTER_HALF)
            & (az > -GATE_OUTER_HALF - GATE_STAND_LENGTH - r)
            & (ax < GATE_STAND_HALF + r + m)
            & (ay < GATE_STAND_HALF + r + m)
        )
        return (in_slab & in_outer & ~in_inner) | in_stand

    hit_gate = jnp.any(frame_region(0.0), axis=-1)
    c_gate = cfg.w_gate * jnp.sum(frame_region(cfg.gate_margin), axis=-1)
    # Pull towards the center line of the target gate when close to its plane
    qt = q[:, track.target_gate]  # (H, 3)
    near_target = (jnp.abs(qt[:, 0]) < 0.3) & (jnp.abs(qt[:, 1]) < 0.5) & (jnp.abs(qt[:, 2]) < 0.5)
    c_gate += cfg.w_gate_center * near_target * (qt[:, 1] ** 2 + qt[:, 2] ** 2)

    # Arena bounds and ground
    low = track.pos_low.at[2].set(cfg.z_min)
    c_bounds = cfg.w_bounds * jnp.sum(
        jnp.maximum(low - pos, 0.0) ** 2 + jnp.maximum(pos - track.pos_high, 0.0) ** 2, axis=-1
    )
    hit_bounds = (pos[:, 2] < 0.0) | jnp.any(pos > track.pos_high + 0.2, axis=-1)
    hit_bounds |= jnp.any(pos[:, :2] < track.pos_low[:2] - 0.2, axis=-1)

    # Collisions are an indicator cost. Once crashed, every following step counts as crashed.
    crashed = jax.lax.cummax((hit_obs | hit_gate | hit_bounds).astype(jnp.float32))
    c_crash = cfg.w_crash * crashed

    stage = c_track + c_ctrl + c_obs + c_gate + c_bounds + c_crash
    return jnp.sum(stage) + terminal, crashed[-1] > 0


def project_on_window(pos: jax.Array, window: PathWindow) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Project a predicted trajectory (H, 3) onto the path window.

    The search for the closest path point only looks a few points around the previous step's
    projection. This keeps the progress monotonic and prevents jumps to other parts of the path
    that pass close by (e.g. when the same gate is passed twice).

    Returns:
        Progress ``s`` (H,), distance to the path (H,), and index of the closest path point (H,).
    """
    n_search = 10
    n = window.pos.shape[0]

    def step(i_prev: jax.Array, p: jax.Array) -> tuple[jax.Array, tuple]:
        start = jnp.clip(i_prev - 2, 0, n - n_search)
        pts = jax.lax.dynamic_slice(window.pos, (start, 0), (n_search, 3))
        i = start + jnp.argmin(jnp.sum((pts - p) ** 2, axis=-1))
        rel = p - window.pos[i]
        along = jnp.dot(rel, window.tangent[i])
        err = jnp.linalg.norm(rel - along * window.tangent[i])
        return i, (window.s[i] + along, err, i)

    _, (s, err, idx) = jax.lax.scan(step, window.idx0, pos)
    return s, err, idx


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step(
    key: jax.Array,
    U: jax.Array,
    x0: jax.Array,
    u_prev: jax.Array,
    ref_pos: jax.Array,
    ref_vel: jax.Array,
    window: PathWindow,
    track: TrackArrays,
    params: ModelParams,
    dist_acc: jax.Array,
    u_low: jax.Array,
    u_high: jax.Array,
    u_hover: jax.Array,
    cfg: MPPIConfig,
) -> tuple[jax.Array, MPPIInfo]:
    """Run one MPPI iteration and return the updated nominal control sequence.

    Args:
        key: JAX random key.
        U: Nominal control sequence (H, 3), warm-started from the previous step.
        x0: Current state (12,).
        u_prev: Previously applied control (3,).
        ref_pos: Reference positions for the horizon (H, 3).
        ref_vel: Reference velocities for the horizon (H, 3).
        window: Path ahead of the drone for the progress cost.
        track: Track information for the cost function.
        params: Model parameters.
        dist_acc: Estimated disturbance acceleration (3,).
        u_low: Lower control limits (3,).
        u_high: Upper control limits (3,).
        u_hover: Hover control (3,).
        cfg: MPPI settings (static).

    Returns:
        The updated control sequence (H, 3) and diagnostics.
    """
    # 1) Sample perturbations. Sample 0 keeps the nominal sequence, sample 1 tries pure hover.
    eps = sample_noise(key, cfg)
    eps = eps.at[0].set(0.0)
    eps = eps.at[1].set(u_hover[None] - U)
    us = jnp.clip(U[None] + eps, u_low, u_high)
    eps = us - U[None]  # Use the effective noise after clipping

    # 2) Roll out all samples in parallel
    xs = jax.vmap(rollout, in_axes=(None, 0, None, None, None))(
        x0, us, params, dist_acc, cfg.n_substeps
    )

    # 3) Evaluate the costs
    cost, crashed = jax.vmap(trajectory_cost, in_axes=(0, 0, None, None, None, None, None, None))(
        xs, us, u_prev, ref_pos, ref_vel, window, track, cfg
    )
    sigma_inv = 1.0 / jnp.asarray(cfg.noise_sigma) ** 2
    cost += cfg.gamma * cfg.temperature * jnp.sum((U - u_hover) * sigma_inv * eps, axis=(1, 2))

    # 4) Exponential weighting (subtract the minimum for numerical stability)
    min_cost = jnp.min(cost)
    weights = jnp.exp(-(cost - min_cost) / cfg.temperature)
    weights = weights / jnp.sum(weights)
    U_new = jnp.clip(U + jnp.einsum("k,khu->hu", weights, eps), u_low, u_high)

    # Diagnostics
    opt_traj = rollout(x0, U_new, params, dist_acc, cfg.n_substeps)[:, :3]
    info = MPPIInfo(
        vis_rollouts=xs[: cfg.n_vis, :, :3],
        vis_cost=(cost[: cfg.n_vis] - min_cost) / cfg.temperature,
        vis_crashed=crashed[: cfg.n_vis],
        opt_traj=opt_traj,
        min_cost=min_cost,
        ess=1.0 / jnp.sum(weights**2),
        crash_ratio=jnp.mean(crashed),
    )
    return U_new, info


@partial(jax.jit, static_argnames=("n_sub",))
def _predict(
    x: jax.Array, u: jax.Array, params: ModelParams, dist_acc: jax.Array, n_sub: int
) -> jax.Array:
    """One-step prediction used by the disturbance observer."""
    return dynamics_step(x, u, params, dist_acc, n_sub)


# region Path planning


class RacingPath:
    """Cubic spline through a list of waypoints, resampled by arc length.

    The path provides positions, unit tangents, and a speed profile as functions of the arc length
    ``s``. All queries use linear interpolation of a dense resampling (~1 cm resolution).
    """

    def __init__(self, waypoints: NDArray, start_tangent: NDArray | None, cfg: MPPIConfig):
        """Fit the spline and compute the speed profile.

        Args:
            waypoints: Waypoints (N, 3).
            start_tangent: Optional unit tangent at the start (e.g. the current flight direction).
            cfg: Settings with the speed limits.
        """
        self.waypoints = waypoints
        chord = np.linalg.norm(np.diff(waypoints, axis=0), axis=-1)
        self.t_knots = np.concatenate([[0.0], np.cumsum(chord)])
        bc = "natural" if start_tangent is None else ((1, start_tangent), (2, np.zeros(3)))
        self.spline = CubicSpline(self.t_knots, waypoints, bc_type=bc)

        n = max(200, int(self.t_knots[-1] / 0.01))
        self.t_dense = np.linspace(0.0, self.t_knots[-1], n)
        self.points = self.spline(self.t_dense)
        ds = np.linalg.norm(np.diff(self.points, axis=0), axis=-1)
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.length = self.s[-1]
        d1, d2 = self.spline(self.t_dense, 1), self.spline(self.t_dense, 2)
        speed = np.linalg.norm(d1, axis=-1) + 1e-9
        self.tangents = d1 / speed[:, None]
        curvature = np.linalg.norm(np.cross(d1, d2), axis=-1) / speed**3

        self._ds, self._a_max = ds, cfg.a_max
        v_curvature = np.sqrt(cfg.a_lat / np.maximum(curvature, 1e-6))
        self._v_curvature = np.clip(v_curvature, cfg.v_min, cfg.v_max)
        self.gate_s: list[float] = []  # Arc length of the gate centers, set by the planner
        self.set_speed_limits([])

    def set_speed_limits(self, limits: list[tuple[float, float, float]]):
        """Compute the speed profile from the curvature limit and additional limits.

        Args:
            limits: List of (s_start, s_end, v_max) sections with a reduced maximum speed.
        """
        v = self._v_curvature.copy()
        for s_lo, s_hi, v_lim in limits:
            v[(self.s >= s_lo) & (self.s <= s_hi)] = np.minimum(
                v[(self.s >= s_lo) & (self.s <= s_hi)], v_lim
            )
        # Backward pass: start braking early enough to respect the deceleration limit
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self._a_max * self._ds[i]))
        self.speed = v

    def position(self, s: NDArray) -> NDArray:
        """Positions at arc lengths ``s``."""
        s = np.clip(s, 0.0, self.length)
        return np.stack([np.interp(s, self.s, self.points[:, i]) for i in range(3)], axis=-1)

    def tangent(self, s: NDArray) -> NDArray:
        """Unit tangents at arc lengths ``s``."""
        s = np.clip(s, 0.0, self.length)
        t = np.stack([np.interp(s, self.s, self.tangents[:, i]) for i in range(3)], axis=-1)
        return t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-9)

    def project(self, pos: NDArray, s_guess: float, window: tuple[float, float]) -> float:
        """Closest point on the path within a window around the last known progress ``s_guess``.

        The window prevents jumps to other parts of the path, e.g. when the same gate is passed
        twice.
        """
        lo = np.searchsorted(self.s, s_guess + window[0])
        hi = max(np.searchsorted(self.s, s_guess + window[1]), lo + 1)
        d = np.linalg.norm(self.points[lo:hi] - pos, axis=-1)
        return float(self.s[lo + np.argmin(d)])

    def reference(
        self, s0: float, v0: float, horizon: int, dt: float, a_max: float
    ) -> tuple[NDArray, NDArray]:
        """Move a reference point along the path, starting at ``s0`` with speed ``v0``.

        The reference accelerates with at most ``a_max`` and never exceeds the speed profile.

        Returns:
            Reference positions and velocities, both (horizon, 3), for times dt, 2 dt, ...
        """
        s_ref, v_ref = np.empty(horizon), np.empty(horizon)
        s, v = s0, v0
        for i in range(horizon):
            v = min(v + a_max * dt, np.interp(s, self.s, self.speed))
            s = s + v * dt
            s_ref[i], v_ref[i] = s, v
        return self.position(s_ref), self.tangent(s_ref) * v_ref[:, None]


@dataclass
class Pole:
    """Vertical cylinder used by the planner to keep the path away from obstacles."""

    xy: NDArray
    z_low: float
    z_high: float
    clearance: float


def gate_normals(gates_quat: NDArray) -> NDArray:
    """Gate x-axes (passing directions) in the world frame."""
    return R.from_quat(gates_quat).apply(np.array([1.0, 0.0, 0.0]))


class GridPlanner:
    """2D shortest path search (Dijkstra) on a grid around obstacles and gate frames.

    The grid planner only routes the drone *between* gates. Every cell has a traversal cost that
    grows close to obstacles, gate frames and the arena boundary, so the shortest path keeps a
    safety distance wherever the track allows it, and goes around clusters of obstacles instead of
    squeezing through narrow gaps. The resulting cell path is simplified to a few corner points.
    """

    def __init__(self, low: tuple[float, float], high: tuple[float, float], res: float):
        """Create the grid and its (static) 8-connected neighborhood graph.

        Args:
            low: Lower xy corner of the arena.
            high: Upper xy corner of the arena.
            res: Grid resolution [m].
        """
        self.res, self.low = res, np.asarray(low)
        xs = np.arange(low[0], high[0] + 1e-6, res)
        ys = np.arange(low[1], high[1] + 1e-6, res)
        self.shape = (len(xs), len(ys))
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.cells = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # Index = ix * ny + iy
        nx, ny = self.shape
        idx = np.arange(nx * ny).reshape(nx, ny)
        src, dst = [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a = idx[0 : nx - dx, max(0, -dy) : ny - max(0, dy)]
            b = idx[dx:nx, max(0, dy) : ny - max(0, -dy)]
            src.append(a.ravel())
            dst.append(b.ravel())
        self.src, self.dst = np.concatenate(src), np.concatenate(dst)
        self.edge_len = np.linalg.norm(self.cells[self.src] - self.cells[self.dst], axis=-1)

    def cell_index(self, xy: NDArray) -> NDArray:
        """Index of the cells closest to the points ``xy`` (..., 2)."""
        ij = np.round((np.asarray(xy) - self.low) / self.res).astype(int)
        ij[..., 0] = np.clip(ij[..., 0], 0, self.shape[0] - 1)
        ij[..., 1] = np.clip(ij[..., 1], 0, self.shape[1] - 1)
        return ij[..., 0] * self.shape[1] + ij[..., 1]

    def cost_field(
        self,
        obstacles_xy: NDArray,
        gate_segments: NDArray,
        gate_clearance: NDArray,
        cfg: MPPIConfig,
    ) -> NDArray:
        """Traversal cost per cell. 1 in free space, higher close to objects, ~1e3 if blocked.

        Args:
            obstacles_xy: Obstacle positions (O, 2).
            gate_segments: Gate frames as xy line segments (G, 2, 2).
            gate_clearance: Distance to each gate frame below which cells are blocked (G,).
            cfg: Planner settings.
        """
        c = self.cells
        d_obs = np.min(np.linalg.norm(c[:, None] - obstacles_xy[None], axis=-1), axis=-1)
        a, b = gate_segments[None, :, 0], gate_segments[None, :, 1]  # (1, G, 2)
        t = np.clip(np.sum((c[:, None] - a) * (b - a), -1) / np.sum((b - a) ** 2, -1), 0, 1)
        d_gates = np.linalg.norm(c[:, None] - (a + t[..., None] * (b - a)), axis=-1)  # (N, G)
        d_gate = np.min(d_gates, axis=-1)
        d_bound = np.min(np.concatenate([c - c.min(0), c.max(0) - c], axis=-1), axis=-1)
        cost = 1.0 + cfg.grid_w_clear * (
            (np.maximum(cfg.grid_r_soft - d_obs, 0.0) / cfg.grid_r_soft) ** 2
            + (np.maximum(cfg.grid_r_soft - d_gate, 0.0) / cfg.grid_r_soft) ** 2
            + (np.maximum(0.3 - d_bound, 0.0) / 0.3) ** 2
        )
        blocked = (d_obs < cfg.obs_clearance) | np.any(d_gates < gate_clearance, axis=-1)
        return cost + 1e3 * blocked

    def graph(self, cost: NDArray) -> csr_matrix:
        """Sparse graph whose edge weights are the edge length times the mean cell cost."""
        weights = self.edge_len * 0.5 * (cost[self.src] + cost[self.dst])
        n = len(self.cells)
        return csr_matrix((weights, (self.src, self.dst)), shape=(n, n))

    def shortest_path(
        self, start: NDArray, goal: NDArray, cost: NDArray, graph: csr_matrix
    ) -> NDArray:
        """Shortest xy path from ``start`` to ``goal``, simplified to its corner points.

        Args:
            start: Start position (2,).
            goal: Goal position (2,).
            cost: Cell costs from ``cost_field``.
            graph: Graph built from the same costs with ``graph``.

        Returns:
            Points (M, 2) including start and goal.
        """
        i_start, i_goal = self.cell_index(start), self.cell_index(goal)
        _, pred = dijkstra(graph, directed=False, indices=i_start, return_predecessors=True)
        cells = [i_goal]
        while cells[-1] != i_start and pred[cells[-1]] >= 0:
            cells.append(pred[cells[-1]])
        if cells[-1] != i_start:  # No path found, fall back to a straight line
            return np.stack([start, goal])
        cells = np.array(cells[::-1])
        points = self.cells[cells].copy()
        points[0], points[-1] = start, goal
        return self._simplify(points, cost[cells], cost)

    def _simplify(self, points: NDArray, path_cost: NDArray, cost: NDArray) -> NDArray:
        """Greedy line-of-sight shortcutting.

        A shortcut from point i to j is accepted if it does not cross cells that are more
        expensive than the most expensive cell of the original path between i and j.
        """
        keep, i = [0], 0
        while i < len(points) - 1:
            j_best = i + 1
            for j in range(i + 2, len(points)):
                n = max(2, int(np.linalg.norm(points[j] - points[i]) / (0.5 * self.res)))
                line = points[i] + np.linspace(0, 1, n)[:, None] * (points[j] - points[i])
                if np.max(cost[self.cell_index(line)]) > np.max(path_cost[i : j + 1]) + 1e-6:
                    break
                j_best = j
            keep.append(j_best)
            i = j_best
        return points[keep]


@lru_cache(maxsize=4)
def _grid_planner(low: tuple[float, float], high: tuple[float, float], res: float) -> GridPlanner:
    """Cached grid planner (the neighborhood graph only depends on the arena size)."""
    return GridPlanner(low, high, res)


def gate_segments(gates_pos: NDArray, gates_quat: NDArray) -> NDArray:
    """Gate frames projected onto the xy plane as line segments (G, 2, 2)."""
    lateral = R.from_quat(gates_quat).apply(np.array([0.0, 1.0, 0.0]))[:, :2]
    lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True) + 1e-9
    return np.stack(
        [
            gates_pos[:, :2] - GATE_OUTER_HALF * lateral,
            gates_pos[:, :2] + GATE_OUTER_HALF * lateral,
        ],
        axis=1,
    )


def plan_path(
    pos: NDArray,
    vel: NDArray,
    gates_pos: NDArray,
    gates_quat: NDArray,
    obstacles_pos: NDArray,
    gate_sequence: NDArray,
    gate_directions: NDArray,
    start_idx: int,
    cfg: MPPIConfig,
    gates_visited: NDArray | None = None,
    bounds: tuple[NDArray, NDArray] = (np.array([-2.5, -1.5]), np.array([2.5, 1.5])),
) -> RacingPath:
    """Plan a smooth path from the current position through all remaining gates.

    Every gate pass is a straight segment from an approach waypoint in front of the gate through
    its center to an exit waypoint behind it. The connections between these segments are found
    with a grid search around obstacles and gate frames. Finally, a cubic spline is fitted through
    all waypoints.

    Args:
        pos: Current drone position.
        vel: Current drone velocity.
        gates_pos: Gate positions (G, 3).
        gates_quat: Gate orientations (G, 4) as xyzw quaternions.
        obstacles_pos: Obstacle top positions (O, 3).
        gate_sequence: Gate IDs in the order they have to be passed.
        gate_directions: Passing directions (1 forward, -1 reverse) of the sequence entries.
        start_idx: Index of the next gate pass in the sequence.
        cfg: Planner settings.
        gates_visited: Flags whether the exact gate poses are known. Used to slow down in front of
            gates whose pose may still change.
        bounds: Lower and upper xy corner of the arena.

    Returns:
        The planned path.
    """
    grid = _grid_planner(tuple(bounds[0][:2]), tuple(bounds[1][:2]), cfg.grid_res)
    segments = gate_segments(gates_pos, gates_quat)
    normals = gate_normals(gates_quat)

    waypoints, fixed = [pos.copy()], [True]
    if pos[2] < 0.1:  # Take off vertically before heading to the first gate
        waypoints.append(pos + np.array([0.0, 0.0, 0.3]))
        fixed.append(False)

    def connect(target: NDArray, thin_gates: list[int]):
        """Append grid waypoints from the last waypoint to ``target`` (z is interpolated).

        The gates we come from and go to only block a thin band around their frame line. Their
        exit and approach waypoints can be very close to the frame if obstacles are in the way.
        """
        start = waypoints[-1]
        gate_clearance = np.full(len(segments), cfg.gate_frame_clearance)
        gate_clearance[thin_gates] = 0.08
        cost = grid.cost_field(obstacles_pos[:, :2], segments, gate_clearance, cfg)
        xy = grid.shortest_path(start[:2], target[:2], cost, grid.graph(cost))
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))])
        z = start[2] + (target[2] - start[2]) * dist / max(dist[-1], 1e-6)
        for p_xy, p_z in zip(xy[1:-1], z[1:-1]):
            # Corners right next to the start or target would create kinks in the spline
            if min(np.linalg.norm(p_xy - start[:2]), np.linalg.norm(p_xy - target[:2])) < 0.15:
                continue
            waypoints.append(np.array([*p_xy, p_z]))
            fixed.append(False)

    def clearance(center: NDArray, d: NDArray, dist: float) -> NDArray:
        """Clearance of the points ``center + t * d``, t in [0, dist], to obstacles on that side."""
        t = np.linspace(0.0, dist, 30)
        side = (obstacles_pos[:, :2] - center[:2]) @ d[:2] > 0.0
        if not np.any(side):
            return np.full_like(t, np.inf)
        pts = center[:2] + t[:, None] * d[:2]
        return np.linalg.norm(pts[:, None] - obstacles_pos[None, side, :2], axis=-1).min(axis=-1)

    def along(center: NDArray, d: NDArray, dist: float) -> NDArray:
        """Point at ``dist`` along ``d`` from the gate center.

        The distance is shortened so that the point stays inside the arena and the straight segment
        from the gate center keeps a minimum clearance to obstacles.
        """
        low, high = bounds[0][:2] + 0.2, bounds[1][:2] - 0.2
        with np.errstate(divide="ignore", invalid="ignore"):
            t_max = np.where(d[:2] > 0, (high - center[:2]) / d[:2], (low - center[:2]) / d[:2])
        t_max = np.where(np.abs(d[:2]) > 1e-6, t_max, np.inf)
        dist = min(dist, np.min(t_max))
        t = np.linspace(0.0, dist, 30)
        if np.any(blocked := clearance(center, d, dist) < cfg.gate_axis_min_clearance):
            t = t[: max(int(np.argmax(blocked)) - 1, 1)]
        return center + max(t[-1], 0.12) * d

    def pass_direction(center: NDArray, n: NDArray) -> NDArray:
        """Direction to fly through a gate.

        Usually the gate normal. If an obstacle stands right in front of or behind the gate, we
        cross the gate diagonally (up to 40 deg) so that approach and exit avoid the obstacle.
        """
        best, best_clearance = n, -1.0
        for angle in (0.0, 0.25, -0.25, 0.45, -0.45, 0.7, -0.7):
            c, s = np.cos(angle), np.sin(angle)
            d = np.array([c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]])
            d_clear = min(
                clearance(center, d, cfg.d_post).min(), clearance(center, -d, cfg.d_pre).min()
            )
            if d_clear >= cfg.gate_axis_clearance:
                return d
            if d_clear > best_clearance:
                best, best_clearance = d, d_clear
        return best

    for k in range(start_idx, len(gate_sequence)):
        gid = gate_sequence[k]
        center = gates_pos[gid]
        d = pass_direction(center, normals[gid] * gate_directions[k])
        pre = along(center, -d, cfg.d_pre)
        # Skip the approach waypoint only if we are already lined up in front of the gate, either
        # along the (possibly tilted) passing direction or along the gate normal
        rel, lined_up = waypoints[-1] - center, False
        for axis in (d, normals[gid] * gate_directions[k]):
            dist_along, lateral = np.dot(rel, axis), np.linalg.norm(rel - np.dot(rel, axis) * axis)
            lined_up |= -(cfg.d_pre + 0.1) < dist_along < 0.0 and lateral < 0.15
        if k != start_idx or not lined_up:
            connect(pre, [gid] + ([gate_sequence[k - 1]] if k > 0 else []))
            waypoints.append(pre)
            fixed.append(True)
        waypoints.append(center.copy())
        fixed.append(True)
        waypoints.append(along(center, d, cfg.d_post))
        fixed.append(True)
    # Extend the path behind the last gate so the reference keeps moving through it
    waypoints.append(along(center, d, cfg.d_post + 0.5))
    fixed.append(True)
    waypoints, fixed = _remove_duplicates(np.array(waypoints), fixed)

    speed = np.linalg.norm(vel)
    tangent = vel / speed if speed > 0.3 else None
    path = RacingPath(waypoints, tangent, cfg)
    # Safety net: the spline may cut corners close to obstacles. Push it away if necessary.
    poles = [Pole(o[:2], 0.0, o[2], cfg.r_obs_safe) for o in obstacles_pos]
    for _ in range(2):
        waypoints, fixed, changed = _push_away_from_poles(path, waypoints, fixed, poles)
        if not changed:
            break
        path = RacingPath(waypoints, tangent, cfg)

    # Locate the gate centers on the path and slow down in front of gates whose exact pose has not
    # been observed yet. Their pose may still change once they come into sensor range.
    i_start, limits = 0, []
    for k in range(start_idx, len(gate_sequence)):
        dist = np.linalg.norm(path.points[i_start:] - gates_pos[gate_sequence[k]], axis=-1)
        i_start += int(np.argmax(dist < 0.03)) if np.any(dist < 0.03) else int(np.argmin(dist))
        path.gate_s.append(float(path.s[i_start]))
        if gates_visited is not None and not gates_visited[gate_sequence[k]]:
            s_gate = path.s[i_start]
            limits.append((s_gate - cfg.unseen_gate_window, s_gate + 0.1, cfg.v_unseen_gate))
    path.set_speed_limits(limits)
    return path


def _remove_duplicates(waypoints: NDArray, fixed: list[bool]) -> tuple[NDArray, list[bool]]:
    """Remove waypoints too close to their predecessor (the spline needs distinct knots)."""
    keep = [0]
    for i in range(1, len(waypoints)):
        if np.linalg.norm(waypoints[i] - waypoints[keep[-1]]) > 0.05:
            keep.append(i)
    return waypoints[keep], [fixed[i] for i in keep]


def _push_away_from_poles(
    path: RacingPath, waypoints: NDArray, fixed: list[bool], poles: list[Pole]
) -> tuple[NDArray, list[bool], bool]:
    """Insert (or move) a detour waypoint for every pole that the path passes too closely."""
    waypoints, fixed, changed = list(waypoints), list(fixed), False
    pts = path.points
    # Segments between two fixed waypoints (gate approach and exit) are not modified
    interval = np.clip(np.searchsorted(path.t_knots, path.t_dense, side="right"), 1, len(fixed) - 1)
    free = ~(np.asarray(fixed)[interval - 1] & np.asarray(fixed)[interval])
    for pole in poles:
        in_height = (pts[:, 2] > pole.z_low - 0.1) & (pts[:, 2] < pole.z_high + 0.1) & free
        d = np.where(in_height, np.linalg.norm(pts[:, :2] - pole.xy, axis=-1), np.inf)
        i = int(np.argmin(d))
        if d[i] >= pole.clearance:
            continue
        new = np.array([*_detour_point(pts[i], path.tangents[i], pole, poles), pts[i, 2]])
        k = int(np.searchsorted(path.t_knots, path.t_dense[i]))  # Insert before knot k
        k = int(np.clip(k, 1, len(waypoints) - 1))
        # Move an existing free waypoint instead of adding a new one right next to it
        near = [j for j in (k - 1, k) if not fixed[j] and np.linalg.norm(waypoints[j] - new) < 0.15]
        if near:
            waypoints[near[0]] = new
        else:
            waypoints.insert(k, new)
            fixed.insert(k, False)
        changed = True
    wps, fixed = _remove_duplicates(np.array(waypoints), fixed)
    return wps, fixed, changed


def _detour_point(point: NDArray, tangent: NDArray, pole: Pole, poles: list[Pole]) -> NDArray:
    """Find the closest xy point next to ``pole`` that keeps the clearance to all poles.

    We try to pass the pole radially outwards and on both sides perpendicular to the path, and
    increase the distance until no other pole is too close. This prevents paths that squeeze
    between two poles that are close to each other.
    """
    perp = np.array([-tangent[1], tangent[0]])
    perp /= np.linalg.norm(perp) + 1e-9
    radial = point[:2] - pole.xy
    radial = radial / n if (n := np.linalg.norm(radial)) > 1e-3 else perp
    others = [o for o in poles if o is not pole and o.z_low - 0.1 < point[2] < o.z_high + 0.1]
    best, best_dev = pole.xy + radial * (pole.clearance + 0.05), np.inf
    for direction in (radial, perp, -perp):
        for r in np.arange(pole.clearance + 0.05, 1.2, 0.05):
            candidate = pole.xy + direction * r
            if all(np.linalg.norm(candidate - o.xy) > o.clearance + 0.03 for o in others):
                if (dev := np.linalg.norm(candidate - point[:2])) < best_dev:
                    best, best_dev = candidate, dev
                break
    return best


# region Controller


class MPPIController(Controller):
    """MPPI controller using the collective thrust and attitude interface."""

    #: Overrides for the default [MPPIConfig][lsy_drone_racing.control.mppi_controller.MPPIConfig]
    #: values, e.g. for parameter sweeps in scripts.
    config_overrides: dict = {}

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller, plan the first path and compile the MPPI step.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError(
                'The MPPI controller outputs attitude commands. Set `control_mode = "attitude"` '
                "in the [env] section of your config."
            )
        self.cfg = MPPIConfig(**{"dt": 1.0 / config.env.freq, **self.config_overrides})
        cfg = self.cfg

        # Model of the drone. The mass is randomized in level 1+; the observer compensates for it.
        p = load_params(Dynamics.so_rpy, config.sim.drone)
        self._params = build_model_params(p, cfg.dt / cfg.n_substeps)
        hover_thrust = (p["mass"] * -p["gravity_vec"][-1] - p["acc_coef"]) / p["cmd_f_coef"]
        self._u_hover = np.array([0.0, 0.0, hover_thrust], dtype=np.float32)
        self._u_low = np.array([-cfg.max_tilt, -cfg.max_tilt, 4 * p["thrust_min"]], np.float32)
        self._u_high = np.array([cfg.max_tilt, cfg.max_tilt, 4 * p["thrust_max"]], np.float32)

        limits = config.env.track.safety_limits
        self._arena_low = np.array(limits.pos_limit_low[:2], dtype=np.float64)
        self._arena_high = np.array(limits.pos_limit_high[:2], dtype=np.float64)
        self._pos_low = np.array(limits.pos_limit_low, dtype=np.float32) + 0.2
        self._pos_high = np.array(limits.pos_limit_high, dtype=np.float32) - 0.2

        self._gate_sequence = np.asarray(obs["gate_sequence"]).reshape(-1)
        self._gate_directions = np.asarray(obs["gate_sequence_direction"]).reshape(-1)
        seed = config.env.seed if config.env.seed > 0 else int(np.random.randint(2**31))
        self._key = jax.random.PRNGKey(seed)

        self._U = np.tile(self._u_hover, (cfg.horizon, 1))
        self._u_prev = self._u_hover.copy()
        self._dist_acc = np.zeros(3, dtype=np.float32)
        self._last_x, self._last_u = None, None
        self._s = 0.0
        self._ref_pos = None
        self._window: PathWindow | None = None
        self._info: MPPIInfo | None = None
        self._tick = 0
        self._n_passed = int(obs["n_gates_passed"])
        self._finished = False
        self._n_replans = 0
        self._solve_times: list[float] = []

        self._track_obs = None
        self._replan(obs)
        self._warmup(obs)

    # --- Helpers ---

    @staticmethod
    def _state(obs: dict[str, NDArray]) -> NDArray:
        """Model state [pos, vel, rpy, rpy_rates] from an observation."""
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        return np.concatenate([obs["pos"], obs["vel"], rpy, drpy]).astype(np.float32)

    def _track_changed(self, obs: dict[str, NDArray]) -> bool:
        """Check if the observed gate or obstacle poses (or their visibility) changed."""
        keys = ("gates_pos", "gates_quat", "obstacles_pos", "gates_visited", "obstacles_visited")
        new = np.concatenate([np.asarray(obs[k], dtype=np.float64).ravel() for k in keys])
        changed = self._track_obs is None or np.max(np.abs(new - self._track_obs)) > 1e-4
        self._track_obs = new
        return changed

    def _replan(self, obs: dict[str, NDArray]):
        """Plan a new path from the current state through all remaining gates."""
        self._track_changed(obs)  # Store the track the plan is based on
        self._gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        self._gates_quat = np.asarray(obs["gates_quat"], dtype=np.float32)
        self._obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        start_idx = int(obs["n_gates_passed"])
        self._path = plan_path(
            np.asarray(obs["pos"], dtype=np.float64),
            np.asarray(obs["vel"], dtype=np.float64),
            self._gates_pos.astype(np.float64),
            self._gates_quat.astype(np.float64),
            self._obstacles_pos.astype(np.float64),
            self._gate_sequence,
            self._gate_directions,
            start_idx,
            self.cfg,
            np.asarray(obs["gates_visited"]).reshape(-1),
            (self._arena_low, self._arena_high),
        )
        self._s = 0.0
        self._path_start_idx = start_idx
        self._n_replans += 1

    def _missed_gate(self, n_passed: int) -> bool:
        """Check if we flew past the target gate on the path without passing it."""
        i = n_passed - self._path_start_idx
        return 0 <= i < len(self._path.gate_s) and self._s > self._path.gate_s[i] + 0.3

    def _track_arrays(self, target: int) -> TrackArrays:
        return TrackArrays(
            gates_pos=jnp.asarray(self._gates_pos),
            gates_rot=jnp.asarray(R.from_quat(self._gates_quat).as_matrix(), dtype=jnp.float32),
            obstacles_pos=jnp.asarray(self._obstacles_pos),
            target_gate=jnp.asarray(target, dtype=jnp.int32),
            pos_low=jnp.asarray(self._pos_low),
            pos_high=jnp.asarray(self._pos_high),
        )

    def _path_window(self) -> PathWindow:
        """The part of the path ahead of the drone that MPPI sees in the progress cost.

        The corridor narrows close to gates (the opening is small) and close to obstacles. The speed
        limit comes from the planner's speed profile, which also knows about curves beyond the
        horizon.
        """
        cfg = self.cfg
        spacing = (cfg.v_max * cfg.horizon * cfg.dt + 0.8) / (cfg.n_window - 1)
        n_back = int(round(0.3 / spacing))
        s = self._s + (np.arange(cfg.n_window) - n_back) * spacing
        pos = self._path.position(s)
        d_gate = np.linalg.norm(pos[:, None] - self._gates_pos[None], axis=-1).min(axis=-1)
        d_obs = np.linalg.norm(pos[:, None, :2] - self._obstacles_pos[None, :, :2], axis=-1)
        width = np.minimum(d_gate - 0.15, d_obs.min(axis=-1) - cfg.r_obs_safe)
        v_max = np.interp(np.clip(s, 0.0, self._path.length), self._path.s, self._path.speed)
        return PathWindow(
            pos=pos.astype(np.float32),
            tangent=self._path.tangent(s).astype(np.float32),
            s=(s - self._s).astype(np.float32),
            width=np.clip(width, 0.05, cfg.corridor_width).astype(np.float32),
            v_max=v_max.astype(np.float32),
            idx0=np.int32(n_back),
        )

    def _warmup(self, obs: dict[str, NDArray]):
        """Compile the jitted functions before the race starts (results are discarded)."""
        t0 = time.perf_counter()
        x0 = self._state(obs)
        ref = np.tile(obs["pos"], (self.cfg.horizon, 1)).astype(np.float32)
        U, _ = mppi_step(
            self._key,
            self._U,
            x0,
            self._u_prev,
            ref,
            np.zeros_like(ref),
            self._path_window(),
            self._track_arrays(0),
            self._params,
            self._dist_acc,
            self._u_low,
            self._u_high,
            self._u_hover,
            cfg=self.cfg,
        )
        jax.block_until_ready(U)
        _predict(x0, self._u_hover, self._params, self._dist_acc, self.cfg.n_substeps)
        logger.debug(f"MPPI compiled in {time.perf_counter() - t0:.2f}s")

    # --- Controller interface ---

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Run one MPPI iteration and return the attitude command.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The attitude command [roll, pitch, yaw, collective thrust] as a numpy array.
        """
        n_passed = int(obs["n_gates_passed"])
        self._n_passed = n_passed
        if n_passed >= len(self._gate_sequence):
            self._finished = True
            return np.array([0.0, 0.0, 0.0, self._u_hover[2]], dtype=np.float32)
        if self._track_changed(obs) or self._missed_gate(n_passed):
            self._replan(obs)

        cfg = self.cfg
        x0 = self._state(obs)
        pos, vel = x0[:3], x0[3:6]
        # Progress along the path and the moving reference point for the horizon
        self._s = self._path.project(pos, self._s, (-0.2, 1.0))
        v0 = max(float(np.dot(vel, self._path.tangent(np.array([self._s]))[0])), 0.0)
        ref_pos, ref_vel = self._path.reference(self._s, v0, cfg.horizon, cfg.dt, cfg.a_max)
        self._ref_pos = ref_pos
        self._window = self._path_window()

        t0 = time.perf_counter()
        self._key, key = jax.random.split(self._key)
        U, self._info = mppi_step(
            key,
            self._U,
            x0,
            self._u_prev,
            ref_pos.astype(np.float32),
            ref_vel.astype(np.float32),
            self._window,
            self._track_arrays(int(self._gate_sequence[n_passed])),
            self._params,
            self._dist_acc,
            self._u_low,
            self._u_high,
            self._u_hover,
            cfg=cfg,
        )
        U = np.asarray(U)
        self._solve_times.append(time.perf_counter() - t0)

        u0 = U[0]
        self._U = np.concatenate([U[1:], U[-1:]], axis=0)  # Warm start for the next step
        self._u_prev = u0
        self._last_x, self._last_u = x0, u0
        return np.array([u0[0], u0[1], 0.0, u0[2]], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Update the disturbance observer with the measured velocity.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        if self._last_x is not None and not terminated:
            x_pred = np.asarray(
                _predict(
                    self._last_x, self._last_u, self._params, self._dist_acc, self.cfg.n_substeps
                )
            )
            residual = (np.asarray(obs["vel"]) - x_pred[3:6]) / self.cfg.dt
            alpha, limit = np.asarray(self.cfg.dist_alpha), np.asarray(self.cfg.dist_limit)
            self._dist_acc = np.clip(self._dist_acc + alpha * residual, -limit, limit)
            self._dist_acc = self._dist_acc.astype(np.float32)
        return self._finished

    def episode_callback(self):
        """Log solver statistics of the episode."""
        if self._solve_times:
            t = np.array(self._solve_times[1:] or self._solve_times) * 1e3
            logger.info(
                f"MPPI: {len(self._solve_times)} steps, solve time mean {t.mean():.1f}ms, "
                f"max {t.max():.1f}ms, replans {self._n_replans}, "
                f"disturbance estimate {np.round(self._dist_acc, 2)} m/s^2"
            )

    def episode_reset(self):
        """Reset the warm start and the observer."""
        self._U = np.tile(self._u_hover, (self.cfg.horizon, 1))
        self._u_prev = self._u_hover.copy()
        self._dist_acc = np.zeros(3, dtype=np.float32)
        self._last_x, self._last_u, self._info = None, None, None
        self._tick, self._finished, self._s = 0, False, 0.0
        self._solve_times = []

    def render_callback(self, sim: Sim):
        """Visualize the planned path, the sampled rollouts and the MPPI solution."""
        cfg = self.cfg
        # Planned path through the gates
        idx = np.linspace(0, len(self._path.points) - 1, 120).astype(int)
        draw_line(
            sim,
            self._path.points[idx],
            rgba=np.array([0.1, 0.8, 0.2, 0.6]),
            start_size=2.0,
            end_size=2.0,
        )
        # Obstacle safety radius used in the cost function, drawn as a band at the drone's height
        z = float(self._ref_pos[0, 2]) if self._ref_pos is not None else 1.0
        for o in self._obstacles_pos:
            draw_capsule(
                sim,
                np.array([o[0], o[1], z - 0.01]),
                np.array([o[0], o[1], z + 0.01]),
                radius=cfg.r_obs_safe,
                rgba=np.array([1.0, 0.5, 0.0, 0.3]),
                cylinder=True,
            )
        # Target gate and passing direction
        if not self._finished:
            n_passed = min(self._n_passed, len(self._gate_sequence) - 1)
            gid = self._gate_sequence[n_passed]
            center = self._gates_pos[gid].astype(np.float64)
            normal = (
                gate_normals(self._gates_quat[gid : gid + 1])[0] * self._gate_directions[n_passed]
            )
            draw_points(sim, center[None], rgba=np.array([1.0, 0.0, 1.0, 0.8]), size=0.03)
            draw_line(
                sim,
                np.stack([center, center + 0.3 * normal]),
                rgba=np.array([1.0, 0.0, 1.0, 1.0]),
                start_size=4.0,
                end_size=1.0,
            )
        if self._info is None:
            return
        info = jax.device_get(self._info)
        # Sampled rollouts colored by their weight exp(-cost / lambda)
        for traj, c, crashed in zip(info.vis_rollouts, info.vis_cost, info.vis_crashed):
            pts = traj[:: cfg.vis_stride].astype(np.float64)
            if crashed:
                rgba = np.array([0.6, 0.0, 0.8, 0.5])
            else:
                x = float(np.clip(c / 10.0, 0.0, 1.0))
                rgba = np.array([x, 1.0 - x, 0.0, 0.7])
            draw_line(sim, pts, rgba=rgba, start_size=1.0, end_size=1.0)
        # What MPPI is asked to do: the path window it makes progress along, or the reference
        # points it tracks. Then the resulting MPPI trajectory.
        if cfg.progress_cost:
            pts = self._window.pos[self._window.idx0 :: 4].astype(np.float64)
            draw_points(sim, pts, rgba=np.array([0.2, 0.4, 1.0, 0.9]), size=0.012)
        else:
            draw_points(sim, self._ref_pos[::3], rgba=np.array([0.2, 0.4, 1.0, 0.9]), size=0.012)
        draw_line(
            sim,
            info.opt_traj.astype(np.float64),
            rgba=np.array([1.0, 0.9, 0.0, 1.0]),
            start_size=5.0,
            end_size=5.0,
        )
