"""MPPI controller for drone racing without a path planner.

This controller is a stripped-down sibling of
[mppi_controller][lsy_drone_racing.control.mppi_controller]. It uses the same drone model, the same
sampling and the same collision costs, but no path planner, no grid search and no spline. MPPI only
knows where the gates and obstacles are and in which order and direction the gates have to be
passed.

Instead of following a path, every rollout is scored by its *distance to go* at the end of the
horizon: an estimate of how far the drone still has to fly to pass all remaining gates. The
estimate is computed from the gate geometry alone:

* In front of the target gate, it is the distance to the gate center, where lateral and vertical
  offsets count more than the distance along the gate normal. This forms a funnel that guides the
  drone into the opening along the passing direction.
* Behind the target gate (on the wrong side), the drone has to fly around the frame: the estimate is
  the distance to a point next to the frame, from there to an entry point in front of the gate, and
  from there to the center.
* Rollouts that pass the gate inside the prediction switch to the next gate, so MPPI already plans
  the exit of a gate towards the next one. The distance to go of the next gates is added as an
  offset, which keeps the cost continuous when a gate is passed.

Obstacles are only handled by the collision costs. This works well for single poles, but MPPI only
looks 0.8 s ahead: if obstacles form a wall between the drone and the gate, no short rollout finds
the way around, and the drone gets stuck in front of the wall. This is exactly the problem the
planner of the other controller solves.

Visualization (with rendering enabled): sampled rollouts colored by their weight (green: high,
red: low, purple: predicted collision), the MPPI plan (yellow), the target gate and its passing
direction (magenta), the upcoming gates (smaller magenta spheres), and the obstacle safety radius
(orange rings).

Run with:

    python scripts/sim.py --config level3.toml --controller mppi_no_planner.py --render True
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mppi_controller import (
    GATE_INNER_HALF,
    GATE_OUTER_HALF,
    ModelParams,
    MPPIConfig,
    MPPIInfo,
    TrackArrays,
    _predict,
    build_model_params,
    rollout,
    safety_cost,
    sample_noise,
)

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoPlannerConfig(MPPIConfig):
    """Settings of the planner-free MPPI.

    Model, sampling, collision and observer settings are inherited from
    [MPPIConfig][lsy_drone_racing.control.mppi_controller.MPPIConfig]. The path and planner
    settings of the parent are not used.
    """

    w_progress: float = 300.0  # Cost per meter of distance to go at the end of the horizon
    lateral_weight: float = 3.0  # In front of a gate, lateral offsets count this much more
    entry_dist: float = 0.4  # Entry point in front of the gate, used to fly around from behind [m]
    corner_margin: float = 0.3  # The way around passes this far beyond the outer frame [m]
    speed_limit: float = 3.5  # Soft speed limit [m/s]
    unseen_gate_dist: float = 1.0  # Below this distance to an unobserved gate, use v_unseen_gate
    n_targets: int = 3  # Number of upcoming gates the rollouts know about


class GateTargets(NamedTuple):
    """The next gates in the race sequence (a JAX pytree). Padded by repeating the last gate."""

    pos: jax.Array  # (N, 3) gate centers
    rot: jax.Array  # (N, 3, 3) gate rotations, x-axis along the required passing direction
    offset: jax.Array  # (N,) distance to go after passing the gate (to the end of the list)
    gate_idx: jax.Array  # (N,) gate IDs, used for the gate centering cost
    unseen: jax.Array  # (N,) whether the exact gate pose has not been observed yet
    n_valid: jax.Array  # () number of real (not padded) entries


def gate_potential(p: jax.Array, g: jax.Array, rot: jax.Array, cfg: NoPlannerConfig) -> jax.Array:
    """Estimated distance to go from ``p`` until the gate at ``g`` is passed.

    Args:
        p: Position (3,).
        g: Gate center (3,).
        rot: Gate rotation (3, 3) with the x-axis along the required passing direction.
        cfg: Settings.
    """
    q = rot.T @ (p - g)
    # In front of the gate: a funnel towards the center along the passing direction
    front = jnp.sqrt(q[0] ** 2 + cfg.lateral_weight * (q[1] ** 2 + q[2] ** 2))
    # Behind the gate: around the closer side of the frame to the entry point, then to the center
    side = jnp.where(q[1] >= 0.0, 1.0, -1.0)
    corner = jnp.array([0.0, side * (GATE_OUTER_HALF + cfg.corner_margin), 0.0])
    entry = jnp.array([-cfg.entry_dist, 0.0, 0.0])
    behind = jnp.linalg.norm(q - corner) + jnp.linalg.norm(corner - entry) + cfg.entry_dist
    return jnp.where(q[0] < 0.0, front, behind)


def distance_to_go(
    pos: jax.Array, p0: jax.Array, targets: GateTargets, cfg: NoPlannerConfig
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Follow a predicted trajectory (H, 3) through the gate sequence.

    Whenever the trajectory crosses the target gate plane inside the opening and in the right
    direction, the target switches to the next gate.

    Args:
        pos: Predicted positions (H, 3).
        p0: Current position (3,), the start of the trajectory.
        targets: Upcoming gates.
        cfg: Settings.

    Returns:
        Distance to go (H,), target gate ID (H,), whether the target is unobserved (H,), and the
        distance to the target gate center (H,).
    """
    n = targets.pos.shape[0]
    opening = GATE_INNER_HALF - cfg.r_drone

    def step(carry: tuple[jax.Array, jax.Array], p: jax.Array) -> tuple[tuple, tuple]:
        k, p_prev = carry
        i = jnp.minimum(k, n - 1)
        g, rot = targets.pos[i], targets.rot[i]
        q0, q1 = rot.T @ (p_prev - g), rot.T @ (p - g)
        alpha = q0[0] / (q0[0] - q1[0] - 1e-9)
        yz = q0[1:] + alpha * (q1[1:] - q0[1:])
        passed = (q0[0] < 0) & (q1[0] >= 0) & jnp.all(jnp.abs(yz) < opening) & (k < targets.n_valid)
        k = k + passed.astype(k.dtype)
        i = jnp.minimum(k, n - 1)
        togo = gate_potential(p, targets.pos[i], targets.rot[i], cfg) + targets.offset[i]
        togo = jnp.where(k >= targets.n_valid, 0.0, togo)  # All gates passed: race finished
        dist = jnp.linalg.norm(p - targets.pos[i])
        return (k, p), (togo, targets.gate_idx[i], targets.unseen[i], dist)

    _, out = jax.lax.scan(step, (jnp.array(0, dtype=jnp.int32), p0), pos)
    return out


def rollout_cost(
    xs: jax.Array,
    us: jax.Array,
    x0: jax.Array,
    u_prev: jax.Array,
    targets: GateTargets,
    track: TrackArrays,
    cfg: NoPlannerConfig,
) -> tuple[jax.Array, jax.Array]:
    """Cost of a single rollout: distance to go at the end plus safety and speed costs.

    Returns:
        The total cost and a flag whether the rollout collides.
    """
    pos, vel = xs[:, 0:3], xs[:, 3:6]
    togo, target_gate, unseen, dist = distance_to_go(pos, x0[:3], targets, cfg)
    c_safety, crashed = safety_cost(xs, us, u_prev, track, target_gate, cfg)
    v_lim = jnp.where(unseen & (dist < cfg.unseen_gate_dist), cfg.v_unseen_gate, cfg.speed_limit)
    c_speed = cfg.w_speed * jnp.maximum(jnp.linalg.norm(vel, axis=-1) - v_lim, 0.0) ** 2
    return jnp.sum(c_safety + c_speed) + cfg.w_progress * togo[-1], crashed[-1]


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step(
    key: jax.Array,
    U: jax.Array,
    x0: jax.Array,
    u_prev: jax.Array,
    targets: GateTargets,
    track: TrackArrays,
    params: ModelParams,
    dist_acc: jax.Array,
    u_low: jax.Array,
    u_high: jax.Array,
    u_hover: jax.Array,
    cfg: NoPlannerConfig,
) -> tuple[jax.Array, MPPIInfo]:
    """One MPPI iteration (sample, roll out, evaluate, weight).

    Same algorithm as in the planner version, only the cost function differs.
    """
    eps = sample_noise(key, cfg)
    eps = eps.at[0].set(0.0)  # Keep the nominal sequence
    eps = eps.at[1].set(u_hover[None] - U)  # Try pure hover
    us = jnp.clip(U[None] + eps, u_low, u_high)
    eps = us - U[None]

    xs = jax.vmap(rollout, in_axes=(None, 0, None, None, None))(
        x0, us, params, dist_acc, cfg.n_substeps
    )
    cost, crashed = jax.vmap(rollout_cost, in_axes=(0, 0, None, None, None, None, None))(
        xs, us, x0, u_prev, targets, track, cfg
    )
    sigma_inv = 1.0 / jnp.asarray(cfg.noise_sigma) ** 2
    cost += cfg.gamma * cfg.temperature * jnp.sum((U - u_hover) * sigma_inv * eps, axis=(1, 2))

    min_cost = jnp.min(cost)
    weights = jnp.exp(-(cost - min_cost) / cfg.temperature)
    weights = weights / jnp.sum(weights)
    U_new = jnp.clip(U + jnp.einsum("k,khu->hu", weights, eps), u_low, u_high)

    info = MPPIInfo(
        vis_rollouts=xs[: cfg.n_vis, :, :3],
        vis_cost=(cost[: cfg.n_vis] - min_cost) / cfg.temperature,
        vis_crashed=crashed[: cfg.n_vis],
        opt_traj=rollout(x0, U_new, params, dist_acc, cfg.n_substeps)[:, :3],
        min_cost=min_cost,
        ess=1.0 / jnp.sum(weights**2),
        crash_ratio=jnp.mean(crashed),
    )
    return U_new, info


class MPPINoPlanner(Controller):
    """MPPI controller without a path planner, using the attitude interface."""

    #: Overrides for the default ``NoPlannerConfig`` values, e.g. for parameter sweeps in scripts.
    config_overrides: dict = {}

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller and compile the MPPI step.

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
        self.cfg = NoPlannerConfig(**{"dt": 1.0 / config.env.freq, **self.config_overrides})
        cfg = self.cfg

        p = load_params(Dynamics.so_rpy, config.sim.drone)
        self._params = build_model_params(p, cfg.dt / cfg.n_substeps)
        hover_thrust = (p["mass"] * -p["gravity_vec"][-1] - p["acc_coef"]) / p["cmd_f_coef"]
        self._u_hover = np.array([0.0, 0.0, hover_thrust], dtype=np.float32)
        self._u_low = np.array([-cfg.max_tilt, -cfg.max_tilt, 4 * p["thrust_min"]], np.float32)
        self._u_high = np.array([cfg.max_tilt, cfg.max_tilt, 4 * p["thrust_max"]], np.float32)
        limits = config.env.track.safety_limits
        self._pos_low = np.array(limits.pos_limit_low, dtype=np.float32) + 0.2
        self._pos_high = np.array(limits.pos_limit_high, dtype=np.float32) - 0.2

        self._gate_sequence = np.asarray(obs["gate_sequence"]).reshape(-1)
        self._gate_directions = np.asarray(obs["gate_sequence_direction"]).reshape(-1)
        seed = config.env.seed if config.env.seed > 0 else int(np.random.randint(2**31))
        self._key = jax.random.PRNGKey(seed)

        self._info: MPPIInfo | None = None
        self._obs = obs
        self.episode_reset()
        self._warmup(obs)

    # --- Helpers ---

    @staticmethod
    def _state(obs: dict[str, NDArray]) -> NDArray:
        """Model state [pos, vel, rpy, rpy_rates] from an observation."""
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        return np.concatenate([obs["pos"], obs["vel"], rpy, drpy]).astype(np.float32)

    def _targets(self, obs: dict[str, NDArray]) -> GateTargets:
        """The next gates of the sequence with their passing directions and distance offsets."""
        n_passed, n_total = int(obs["n_gates_passed"]), len(self._gate_sequence)
        k = np.minimum(np.arange(n_passed, n_passed + self.cfg.n_targets), n_total - 1)
        ids, dirs = self._gate_sequence[k], self._gate_directions[k]
        flip = np.stack([dirs, dirs, np.ones_like(dirs)], axis=-1)[:, None, :]  # Reverse passes
        rot = (R.from_quat(obs["gates_quat"][ids]).as_matrix() * flip).astype(np.float32)
        pos = np.asarray(obs["gates_pos"][ids], dtype=np.float32)
        n_valid = min(self.cfg.n_targets, n_total - n_passed)
        # Distance to go after passing gate j: the potentials of the following gates, each
        # evaluated at the center of the gate before. Keeps the cost continuous at a gate pass.
        legs = [
            float(gate_potential(pos[j - 1], pos[j], rot[j], self.cfg)) if j < n_valid else 0.0
            for j in range(1, self.cfg.n_targets)
        ] + [0.0]
        offset = np.cumsum(legs[::-1])[::-1].astype(np.float32)
        return GateTargets(
            pos=pos,
            rot=rot,
            offset=offset,
            gate_idx=ids.astype(np.int32),
            unseen=~np.asarray(obs["gates_visited"])[ids],
            n_valid=np.int32(n_valid),
        )

    def _track_arrays(self, obs: dict[str, NDArray], target: int) -> TrackArrays:
        return TrackArrays(
            gates_pos=jnp.asarray(obs["gates_pos"], dtype=jnp.float32),
            gates_rot=jnp.asarray(R.from_quat(obs["gates_quat"]).as_matrix(), dtype=jnp.float32),
            obstacles_pos=jnp.asarray(obs["obstacles_pos"], dtype=jnp.float32),
            target_gate=jnp.asarray(target, dtype=jnp.int32),
            pos_low=jnp.asarray(self._pos_low),
            pos_high=jnp.asarray(self._pos_high),
        )

    def _mppi(self, key: jax.Array, obs: dict[str, NDArray]) -> tuple[jax.Array, MPPIInfo]:
        targets = self._targets(obs)
        return mppi_step(
            key,
            self._U,
            self._state(obs),
            self._u_prev,
            targets,
            self._track_arrays(obs, int(targets.gate_idx[0])),
            self._params,
            self._dist_acc,
            self._u_low,
            self._u_high,
            self._u_hover,
            cfg=self.cfg,
        )

    def _warmup(self, obs: dict[str, NDArray]):
        """Compile the jitted functions before the race starts (results are discarded)."""
        U, _ = self._mppi(self._key, obs)
        jax.block_until_ready(U)
        x0 = self._state(obs)
        _predict(x0, self._u_hover, self._params, self._dist_acc, self.cfg.n_substeps)

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
        self._obs = obs
        if int(obs["n_gates_passed"]) >= len(self._gate_sequence):
            self._finished = True
            return np.array([0.0, 0.0, 0.0, self._u_hover[2]], dtype=np.float32)

        t0 = time.perf_counter()
        self._key, key = jax.random.split(self._key)
        U, self._info = self._mppi(key, obs)
        U = np.asarray(U)
        self._solve_times.append(time.perf_counter() - t0)

        u0 = U[0]
        self._U = np.concatenate([U[1:], U[-1:]], axis=0)  # Warm start for the next step
        self._u_prev = u0
        self._last_x, self._last_u = self._state(obs), u0
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
            logger.info(f"MPPI (no planner): solve time mean {t.mean():.1f}ms, max {t.max():.1f}ms")

    def episode_reset(self):
        """Reset the warm start and the observer."""
        self._U = np.tile(self._u_hover, (self.cfg.horizon, 1))
        self._u_prev = self._u_hover.copy()
        self._dist_acc = np.zeros(3, dtype=np.float32)
        self._last_x, self._last_u, self._info = None, None, None
        self._finished = False
        self._solve_times: list[float] = []

    def render_callback(self, sim: Sim):
        """Visualize the gate targets, the sampled rollouts and the MPPI solution."""
        cfg, obs = self.cfg, self._obs
        z = float(obs["pos"][2])
        for o in np.asarray(obs["obstacles_pos"], dtype=np.float64):
            draw_capsule(
                sim,
                np.array([o[0], o[1], z - 0.01]),
                np.array([o[0], o[1], z + 0.01]),
                radius=cfg.r_obs_safe,
                rgba=np.array([1.0, 0.5, 0.0, 0.3]),
                cylinder=True,
            )
        if self._finished or self._info is None:
            return
        targets = self._targets(obs)
        center = np.asarray(targets.pos[0], dtype=np.float64)
        normal = np.asarray(targets.rot[0][:, 0], dtype=np.float64)
        draw_points(sim, center[None], rgba=np.array([1.0, 0.0, 1.0, 0.8]), size=0.03)
        draw_line(
            sim,
            np.stack([center - cfg.entry_dist * normal, center + 0.3 * normal]),
            rgba=np.array([1.0, 0.0, 1.0, 1.0]),
            start_size=1.0,
            end_size=4.0,
        )
        n_valid = int(targets.n_valid)
        if n_valid > 1:
            nxt = np.asarray(targets.pos[1:n_valid], dtype=np.float64)
            draw_points(sim, nxt, rgba=np.array([1.0, 0.0, 1.0, 0.5]), size=0.02)
        info = jax.device_get(self._info)
        for traj, c, crashed in zip(info.vis_rollouts, info.vis_cost, info.vis_crashed):
            if crashed:
                rgba = np.array([0.6, 0.0, 0.8, 0.5])
            else:
                x = float(np.clip(c / 10.0, 0.0, 1.0))
                rgba = np.array([x, 1.0 - x, 0.0, 0.7])
            pts = traj[:: cfg.vis_stride].astype(np.float64)
            draw_line(sim, pts, rgba=rgba, start_size=1.0, end_size=1.0)
        draw_line(
            sim,
            info.opt_traj.astype(np.float64),
            rgba=np.array([1.0, 0.9, 0.0, 1.0]),
            start_size=5.0,
            end_size=5.0,
        )
