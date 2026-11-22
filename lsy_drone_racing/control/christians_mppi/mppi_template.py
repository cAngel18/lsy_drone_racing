"""Model Predictive Path Integral (MPPI) controller in JAX for the LSY drone racing course.

Sampling-based MPC as in lecture L3 (slides 29-32). Every control step we

1. sample K perturbed input sequences around the current nominal input sequence,
2. roll all of them out through Crazyflow's identified ``so_rpy`` drone model,
3. score each rollout with a racing cost (progress through the gate sequence, collisions, ...),
4. average the sequences with the weights w_k = exp(-(J_k - J_min) / lambda),
5. apply the first input and shift the sequence by one step (warm start for the next step).

Usage:
    1. Copy this file to ``lsy_drone_racing/control/``.
    2. Set ``control_mode = "attitude"`` in the ``[env]`` section of your level config.
    3. ``python scripts/sim.py --config level0.toml --controller mppi_controller.py``

Implementation notes:
    * The whole MPPI step (sampling, all rollouts, weighting) is one jit-compiled function. It is
      compiled once in ``__init__`` and reused afterwards, also across episodes, as long as
      ``MPPIConfig`` and the array shapes stay the same.
    * Gate and obstacle positions are passed into the jitted function as arguments in every step.
      They change when the drone gets within sensor range, and constants captured from outside a
      jitted function would be frozen at their values from the first call.
    * The rollout model is the Euler-angle form of ``crazyflow.dynamics.so_rpy`` with the
      identified parameters of the course drone. A simple disturbance observer adds a constant
      acceleration bias to compensate for model mismatch (e.g. randomized mass).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_line
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# Collision geometry from lsy_drone_racing/envs/assets/{gate,obstacle}.xml, expressed in the gate
# frame. Gates are passed along their local x-axis, the frame lies in the local y-z plane.
GATE_INNER = 0.2  # Half width of the gate opening [m]
GATE_OUTER = 0.36  # Half width of the outer frame [m]
GATE_HALF_THICKNESS = 0.01  # [m]
STAND_HALF_WIDTH = 0.05  # The stand below each gate [m]
OBSTACLE_RADIUS = 0.015  # Obstacles are thin vertical poles [m]


class MPPIConfig(NamedTuple):
    """All tuning knobs. A NamedTuple is hashable, so it can be a static argument of jax.jit."""

    # Sampling
    n_samples: int = 1024
    horizon: int = 30  # Number of steps of length dt -> 0.6 s lookahead at 50 Hz
    dt: float = 0.02  # Overwritten with 1 / env freq
    temperature: float = 2.0  # lambda in w_k = exp(-(J_k - J_min) / lambda)
    noise_std: tuple[float, float, float, float] = (0.2, 0.2, 0.0, 0.08)  # roll, pitch, yaw, thrust
    noise_corr: float = 0.8  # AR(1) correlation of the noise along the horizon (0 = white noise)
    max_tilt: float = 0.6  # Roll and pitch command limit [rad]
    # Progress cost: remaining race distance, measured through an entry point in front of each gate
    entry_dist: float = 0.4  # Distance of the entry point in front of the gate [m]
    funnel_radius_entry: float = 0.25  # Radius of the approach funnel at the entry point [m]
    funnel_radius_gate: float = 0.1  # Radius of the approach funnel at the gate [m]
    w_progress: float = 1.0
    # Safety and regularization
    drone_radius: float = 0.07  # Used for collision checks [m]
    crash_cost: float = 1000.0  # Paid once per rollout at the first collision
    obstacle_safe_dist: float = 0.25  # Soft clearance to the poles [m]
    w_obstacle: float = 100.0
    gate_safe_dist: float = 0.15  # Soft clearance to the gate frames [m]
    w_gate: float = 100.0
    z_min: float = 0.15  # Soft minimum altitude [m]
    w_ground: float = 100.0
    v_max: float = 2.0  # Soft speed limit [m/s]
    w_speed: float = 10.0
    w_tilt: float = 0.5
    w_smooth: tuple[float, float, float, float] = (5.0, 5.0, 0.0, 20.0)  # On input changes
    # Disturbance observer
    bias_gain: float = 0.1  # Low-pass gain of the acceleration bias estimate
    bias_limit: float = 3.0  # [m/s^2]


# region Model


def body_z_axis(rpy: jax.Array) -> jax.Array:
    """Third column of R = Rz(yaw) Ry(pitch) Rx(roll), i.e. scipy's extrinsic "xyz" convention."""
    cr, sr = jnp.cos(rpy[0]), jnp.sin(rpy[0])
    cp, sp = jnp.cos(rpy[1]), jnp.sin(rpy[1])
    cy, sy = jnp.cos(rpy[2]), jnp.sin(rpy[2])
    return jnp.stack([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])


def so_rpy_step(x: jax.Array, u: jax.Array, p: dict, acc_bias: jax.Array, dt: float) -> jax.Array:
    """One semi-implicit Euler step of the so_rpy model (``crazyflow.dynamics.so_rpy``).

    Args:
        x: State [pos(3), rpy(3), vel(3), rpy_rates(3)].
        u: Command [roll, pitch, yaw, thrust], the env's attitude action format.
        p: Model parameters.
        acc_bias: Estimated unmodelled acceleration [m/s^2].
        dt: Step length [s].
    """
    pos, rpy, vel, rpy_rates = x[0:3], x[3:6], x[6:9], x[9:12]
    rpy_acc = p["rpy_coef"] * rpy + p["rpy_rates_coef"] * rpy_rates + p["cmd_rpy_coef"] * u[:3]
    rpy_rates = rpy_rates + dt * rpy_acc
    rpy = rpy + dt * rpy_rates
    thrust = p["acc_coef"] + p["cmd_f_coef"] * u[3]
    acc = thrust / p["mass"] * body_z_axis(rpy) + p["gravity_vec"] + acc_bias
    vel = vel + dt * acc
    pos = pos + dt * vel
    return jnp.concatenate([pos, rpy, vel, rpy_rates])


# region Cost


def target_gate(k: jax.Array, track: dict) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Center, rotation and passing direction (+1/-1) of the k-th gate in the gate sequence."""
    k = jnp.minimum(k, track["seq"].shape[0] - 1)
    gate_id = track["seq"][k]
    return track["gates_pos"][gate_id], track["gates_rot"][gate_id], track["seq_dir"][k]


def remaining_distance(track: dict, cfg: MPPIConfig) -> jax.Array:
    """rem[k]: Path length from the center of the k-th gate of the sequence to the finish.

    The path goes from each gate center through the entry point of the next gate to its center.
    """
    seq, direction = track["seq"], track["seq_dir"]
    centers = track["gates_pos"][seq]
    normals = track["gates_rot"][seq][:, :, 0] * direction[:, None]  # In passing direction
    entries = centers - cfg.entry_dist * normals
    segments = jnp.linalg.norm(entries[1:] - centers[:-1], axis=-1) + cfg.entry_dist
    rem = jnp.cumsum(segments[::-1])[::-1]
    return jnp.concatenate([rem, jnp.zeros(1)])


def distance_to_go(pos: jax.Array, k: jax.Array, track: dict, cfg: MPPIConfig) -> jax.Array:
    """Remaining race distance when the k-th gate of the sequence is the next target.

    Inside a funnel in front of the gate we measure the distance to the gate center. Outside of
    it, we measure the distance to the entry point in front of the gate plus the distance from the
    entry point to the gate. This keeps the drone from approaching gates from the side, and the
    cost is (almost) continuous when a gate is passed and the target switches to the next one.
    """
    n_passes = track["seq"].shape[0]
    center, rot, direction = target_gate(k, track)
    local = rot.T @ (pos - center)
    x_local, lateral = local[0] * direction, jnp.linalg.norm(local[1:])
    entry = center - cfg.entry_dist * direction * rot[:, 0]
    frac = jnp.clip(-x_local / cfg.entry_dist, 0.0, 1.0)
    radius = cfg.funnel_radius_gate + (cfg.funnel_radius_entry - cfg.funnel_radius_gate) * frac
    in_funnel = (x_local > -cfg.entry_dist) & (x_local <= 0.0) & (lateral < radius)
    to_gate = jnp.where(
        in_funnel, jnp.linalg.norm(pos - center), jnp.linalg.norm(pos - entry) + cfg.entry_dist
    )
    rem = track["rem"][jnp.minimum(k, n_passes - 1)]
    return jnp.where(k >= n_passes, 0.0, to_gate + rem)


def gate_passed(pos_prev: jax.Array, pos: jax.Array, k: jax.Array, track: dict) -> jax.Array:
    """Whether the segment pos_prev -> pos passes the k-th gate of the sequence correctly."""
    center, rot, direction = target_gate(k, track)
    l0, l1 = rot.T @ (pos_prev - center), rot.T @ (pos - center)
    x0, x1 = l0[0] * direction, l1[0] * direction
    crossed = (x0 < 0.0) & (x1 >= 0.0)
    alpha = -x0 / jnp.where(crossed, x1 - x0, 1.0)  # Interpolate the plane intersection
    yz = l0[1:] + alpha * (l1[1:] - l0[1:])
    return crossed & jnp.all(jnp.abs(yz) < GATE_INNER) & (k < track["seq"].shape[0])


def collision(pos: jax.Array, track: dict, cfg: MPPIConfig) -> jax.Array:
    """Collision with any gate frame, gate stand, obstacle pole, or the safety limits."""
    r = cfg.drone_radius
    local = jnp.einsum("gji,gj->gi", track["gates_rot"], pos - track["gates_pos"])  # R^T (p - c)
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


def stage_cost(
    x: jax.Array, u: jax.Array, u_prev: jax.Array, k: jax.Array, track: dict, cfg: MPPIConfig
) -> jax.Array:
    """Smooth part of the running cost. The crash penalty is added in the rollout."""
    pos, vel = x[0:3], x[6:9]
    cost = cfg.w_progress * distance_to_go(pos, k, track, cfg)
    obstacles = track["obstacles_pos"]
    d_pole = jnp.linalg.norm(pos[:2] - obstacles[:, :2], axis=-1)
    below_top = pos[2] < obstacles[:, 2] + cfg.obstacle_safe_dist
    clearance = jnp.maximum(cfg.obstacle_safe_dist - d_pole, 0.0) ** 2
    cost += cfg.w_obstacle * jnp.sum(clearance * below_top)
    # Distance to the square frame of each gate: in the gate plane, then combined with the offset
    local = jnp.einsum("gji,gj->gi", track["gates_rot"], pos - track["gates_pos"])
    square_dist = jnp.maximum(jnp.abs(local[:, 1]), jnp.abs(local[:, 2]))
    in_plane = jnp.maximum(jnp.maximum(GATE_INNER - square_dist, square_dist - GATE_OUTER), 0.0)
    d_frame = jnp.sqrt(local[:, 0] ** 2 + in_plane**2)
    cost += cfg.w_gate * jnp.sum(jnp.maximum(cfg.gate_safe_dist - d_frame, 0.0) ** 2)
    cost += cfg.w_ground * jnp.maximum(cfg.z_min - pos[2], 0.0) ** 2
    cost += cfg.w_speed * jnp.maximum(jnp.linalg.norm(vel) - cfg.v_max, 0.0) ** 2
    cost += cfg.w_tilt * (u[0] ** 2 + u[1] ** 2)
    cost += jnp.sum(jnp.asarray(cfg.w_smooth) * (u - u_prev) ** 2)
    return cost


# region MPPI


def rollout(
    U: jax.Array,
    x0: jax.Array,
    u_prev: jax.Array,
    k0: jax.Array,
    acc_bias: jax.Array,
    params: dict,
    track: dict,
    cfg: MPPIConfig,
) -> tuple[jax.Array, jax.Array]:
    """Simulate one input sequence U (horizon, 4). Returns its total cost and the positions."""

    def step(carry: tuple, u: jax.Array) -> tuple[tuple, tuple]:
        x, k, crashed, u_last = carry
        x_next = so_rpy_step(x, u, params, acc_bias, cfg.dt)
        k_next = k + (gate_passed(x[:3], x_next[:3], k, track) & ~crashed)
        hit = collision(x_next[:3], track, cfg)
        cost = stage_cost(x_next, u, u_last, k_next, track, cfg)
        cost += cfg.crash_cost * (hit & ~crashed)
        return (x_next, k_next, crashed | hit, u), (cost, x_next[:3])

    carry0 = (x0, k0, jnp.array(False), u_prev)
    _, (costs, positions) = jax.lax.scan(step, carry0, U)
    return costs.sum(), positions


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step(
    key: jax.Array,
    U: jax.Array,
    x0: jax.Array,
    u_prev: jax.Array,
    k0: jax.Array,
    acc_bias: jax.Array,
    params: dict,
    track: dict,
    cfg: MPPIConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
    """One MPPI iteration. Returns (key, action, shifted nominal sequence, planned path, stats)."""
    track = dict(track, rem=remaining_distance(track, cfg))
    key, subkey = jax.random.split(key)
    K, N = cfg.n_samples, cfg.horizon

    # Temporally correlated Gaussian noise (AR(1) process) for smoother input sequences
    white = jax.random.normal(subkey, (N, K, 4)) * jnp.asarray(cfg.noise_std)
    beta = cfg.noise_corr

    def ar1(eps: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
        eps = beta * eps + jnp.sqrt(1.0 - beta**2) * w
        return eps, eps

    _, eps = jax.lax.scan(ar1, white[0], white[1:])
    eps = jnp.concatenate([white[:1], eps], axis=0).transpose(1, 0, 2)  # (K, N, 4)
    eps = eps.at[0].set(0.0)  # Sample 0 is the unperturbed nominal sequence

    low = jnp.array([-cfg.max_tilt, -cfg.max_tilt, -jnp.pi, params["thrust_min"]])
    high = jnp.array([cfg.max_tilt, cfg.max_tilt, jnp.pi, params["thrust_max"]])
    samples = jnp.clip(U[None] + eps, low, high)

    # Roll out all samples in parallel: vmap over the samples, lax.scan over the horizon
    rollout_fn = partial(rollout, cfg=cfg)
    batched = jax.vmap(rollout_fn, in_axes=(0, None, None, None, None, None, None))
    costs, _ = batched(samples, x0, u_prev, k0, acc_bias, params, track)

    # Exponential weighting (lecture L3, slide 31). Subtracting J_min keeps exp() from underflowing
    weights = jnp.exp(-(costs - costs.min()) / cfg.temperature)
    weights = weights / weights.sum()
    U_new = jnp.einsum("k,kti->ti", weights, samples)

    _, planned_path = rollout_fn(U_new, x0, u_prev, k0, acc_bias, params, track)
    U_next = jnp.concatenate([U_new[1:], U_new[-1:]], axis=0)  # Shift for the warm start
    stats = {
        "cost_min": costs.min(),
        "cost_nominal": costs[0],
        "effective_samples": 1.0 / jnp.sum(weights**2),  # Use this to tune the temperature
    }
    return key, U_new[0], U_next, planned_path, stats


predict_step = jax.jit(so_rpy_step)


# region Controller


class MPPIController(Controller):
    """MPPI controller that commands collective thrust and attitude."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load the model, set up the nominal input sequence, and compile the MPPI step."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError('MPPIController needs `control_mode = "attitude"` in [env].')
        self.cfg = MPPIConfig(dt=1.0 / config.env.freq)

        p = load_params(Dynamics.so_rpy, config.sim.drone)
        model_keys = ("mass", "gravity_vec", "acc_coef", "cmd_f_coef")
        model_keys += ("rpy_coef", "rpy_rates_coef", "cmd_rpy_coef")
        self.params = {k: jnp.asarray(p[k], dtype=jnp.float32) for k in model_keys}
        self.params["thrust_min"] = jnp.float32(4 * p["thrust_min"])  # Collective thrust limits
        self.params["thrust_max"] = jnp.float32(4 * p["thrust_max"])
        self.hover_thrust = float((p["mass"] * 9.81 - p["acc_coef"]) / p["cmd_f_coef"])

        limits = config.env.track.safety_limits
        self.pos_low = jnp.asarray(limits.pos_limit_low, dtype=jnp.float32)
        self.pos_high = jnp.asarray(limits.pos_limit_high, dtype=jnp.float32)
        seed = config.env.seed if config.env.seed >= 0 else np.random.randint(2**31)
        self._seed = int(seed)
        self.episode_reset()

        # Compile now, not during the first control step of the race. Outputs are discarded.
        mppi_step(
            self._key,
            self._U,
            self._state(obs),
            jnp.asarray(self._u_prev),
            jnp.int32(obs["n_gates_passed"]),
            jnp.asarray(self._acc_bias),
            self.params,
            self._track(obs),
            self.cfg,
        )[1].block_until_ready()
        predict_step(self._state(obs), self._U[0], self.params, self._acc_bias, self.cfg.dt)

    def episode_reset(self):
        """Reset the nominal sequence, random key and disturbance estimate."""
        hover = np.array([0.0, 0.0, 0.0, self.hover_thrust], dtype=np.float32)
        self._U = jnp.tile(jnp.asarray(hover), (self.cfg.horizon, 1))
        self._u_prev = hover
        self._key = jax.random.PRNGKey(self._seed)
        self._acc_bias = np.zeros(3, dtype=np.float32)
        self._x_prev = None
        self._planned_path = None
        self.stats = {}

    def reset(self):
        """Reset internal variables."""
        self.episode_reset()

    def _state(self, obs: dict[str, NDArray[np.floating]]) -> jax.Array:
        """Convert the observation into the model state [pos, rpy, vel, rpy_rates]."""
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        rpy_rates = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x = np.concatenate([obs["pos"], rpy, obs["vel"], rpy_rates])
        return jnp.asarray(x, dtype=jnp.float32)

    def _track(self, obs: dict[str, NDArray[np.floating]]) -> dict[str, jax.Array]:
        """Current (sensed) track layout. Always the same shapes and dtypes to avoid recompiles."""
        return {
            "gates_pos": jnp.asarray(obs["gates_pos"], dtype=jnp.float32),
            "gates_rot": jnp.asarray(R.from_quat(obs["gates_quat"]).as_matrix(), dtype=jnp.float32),
            "seq": jnp.asarray(obs["gate_sequence"], dtype=jnp.int32),
            "seq_dir": jnp.asarray(obs["gate_sequence_direction"], dtype=jnp.float32),
            "obstacles_pos": jnp.asarray(obs["obstacles_pos"], dtype=jnp.float32),
            "pos_low": self.pos_low,
            "pos_high": self.pos_high,
        }

    def _update_disturbance(self, x: jax.Array, obs: dict[str, NDArray[np.floating]]):
        """Compare the predicted and measured velocity and low-pass filter the difference."""
        if self._x_prev is None or obs["pos"][2] < 0.05:  # On the ground, contacts dominate
            return
        u_prev = jnp.asarray(self._u_prev)
        x_pred = predict_step(self._x_prev, u_prev, self.params, self._acc_bias, self.cfg.dt)
        acc_error = (obs["vel"] - np.asarray(x_pred[6:9])) / self.cfg.dt
        self._acc_bias = np.clip(
            self._acc_bias + self.cfg.bias_gain * acc_error,
            -self.cfg.bias_limit,
            self.cfg.bias_limit,
        ).astype(np.float32)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Run one MPPI iteration and return the attitude command [roll, pitch, yaw, thrust]."""
        x = self._state(obs)
        self._update_disturbance(x, obs)
        self._key, action, self._U, path, stats = mppi_step(
            self._key,
            self._U,
            x,
            jnp.asarray(self._u_prev),
            jnp.int32(obs["n_gates_passed"]),
            jnp.asarray(self._acc_bias),
            self.params,
            self._track(obs),
            self.cfg,
        )
        action = np.asarray(action, dtype=np.float32)
        self._x_prev, self._u_prev = x, action
        self._planned_path = path
        self.stats = stats  # Device arrays. Convert with float() only when you log them
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """The base class returns True here, which would end the episode after one step."""
        return False

    def render_callback(self, sim: Sim):
        """Draw the planned path of the current MPPI solution."""
        if self._planned_path is not None:
            draw_line(sim, np.asarray(self._planned_path), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
