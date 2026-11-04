"""Reinforcement-learning racing controller for the attitude interface.

A small MLP maps the race observation directly to a roll, pitch and collective-thrust command at
the environment frequency. The policy is trained with PPO on the level 2 randomisations in
[train_rl_racer][lsy_drone_racing.control.train_rl_racer] (after Kaufmann et al., Nature 2023, and
Song et al., IROS 2021) and sees exactly what this controller sees: gates and obstacles are at their
nominal poses until they enter the sensor range, so reacting to a reveal is part of what it learns.

The observation and action conventions are defined here and are shared with the trainer, which is
why the functions below take the array namespace ``xp`` (``numpy`` here, ``jax.numpy`` there). The
weights are stored in a plain ``.npz`` file, so inference needs neither JAX nor torch.

Status (820 level 2 runs through ``scripts/sim.py`` and ``scripts/evaluate.py``): 61 % finished,
mean lap 3.91 s, std 0.15 s, best 3.58 s. Levels 0 and 1 (100 runs each): all finished, mean laps
3.86 s and 3.88 s. The policy is specific to this track, so it does not apply to level 3. It has
only been flown in simulation and commands tilts of up to 1.5 rad.

Requires ``control_mode = "attitude"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

POLICY_PATH = Path(__file__).parent / "rl_racer_policy.npz"
G = 9.81  # [m/s^2]

# Observation scales
POS_SCALE = 2.0  # [m]
VEL_SCALE = 3.0  # [m/s]
ANG_VEL_SCALE = 10.0  # [rad/s]
MASS_RATIO_SCALE = 0.12  # level 2 randomises the mass by +-11.5 %
# Online estimate of the mass (recursive least squares, T = m (a_meas + g) . z_body)
MASS_PRIOR_N = 25.0  # weight of the nominal mass prior, in equivalent hover samples
MASS_FORGET = 0.995  # forgetting factor per step
MASS_BOUNDS = (0.85, 1.15)  # bounds relative to the nominal mass
MASS_MIN_HEIGHT = 0.08  # [m] no updates close to the ground
MASS_THRUST_BOUNDS = (0.15, 0.85)  # no updates when the thrust is saturated, relative to its range
THRUST_LAG = 0.55  # per step, first-order lag from the commanded to the realised thrust


def quat_to_matrix(xp: Any, q: Any) -> Any:
    """Rotation matrices (..., 3, 3) from xyzw quaternions (..., 4)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ]
    return xp.stack(rows, axis=-1).reshape(q.shape[:-1] + (3, 3))


def action_to_command(xp: Any, action: Any, limits: dict[str, Any]) -> Any:
    """Map policy actions in [-1, 1]^3 (roll, pitch, thrust) to [roll, pitch, yaw, thrust]."""
    a = xp.clip(action, -1.0, 1.0)
    thrust = limits["thrust_mid"] + a[..., 2] * limits["thrust_half"]
    yaw = xp.zeros_like(thrust)  # Yaw is held at zero, nothing on this track needs it
    return xp.stack(
        [a[..., 0] * limits["max_tilt"], a[..., 1] * limits["max_tilt"], yaw, thrust], axis=-1
    )


def mass_estimate_init(xp: Any, n: int, mass_nominal: float) -> dict[str, Any]:
    """Initial state of the mass estimator for n drones."""
    prior = MASS_PRIOR_N * G * G
    return {
        "num": xp.full((n,), prior * mass_nominal, dtype=xp.float32),
        "den": xp.full((n,), prior, dtype=xp.float32),
        "thrust": xp.full((n,), mass_nominal * G, dtype=xp.float32),
    }


def mass_estimate_update(
    xp: Any,
    est: dict[str, Any],
    pos: Any,
    quat: Any,
    vel: Any,
    vel_prev: Any,
    thrust_cmd: Any,
    limits: dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    """One least-squares update of the mass estimate from the measured acceleration.

    Args:
        xp: The array namespace.
        est: The estimator state.
        pos: Current positions (n, 3).
        quat: Current orientations (n, 4).
        vel: Current velocities (n, 3).
        vel_prev: Velocities one step ago (n, 3).
        thrust_cmd: Thrust commanded during the last step (n,).
        limits: The action limits.
        dt: The step duration.

    Returns:
        The updated estimator state.
    """
    thrust = est["thrust"] + THRUST_LAG * (thrust_cmd - est["thrust"])
    z_body = quat_to_matrix(xp, quat)[..., 2]
    acc = (vel - vel_prev) / dt
    y = xp.sum(acc * z_body, axis=-1) + G * z_body[..., 2]
    t_low = limits["thrust_mid"] - limits["thrust_half"]
    t_range = 2 * limits["thrust_half"]
    active = (
        (pos[..., 2] > MASS_MIN_HEIGHT)
        & (thrust > t_low + MASS_THRUST_BOUNDS[0] * t_range)
        & (thrust < t_low + MASS_THRUST_BOUNDS[1] * t_range)
    )
    num = xp.where(active, MASS_FORGET * est["num"] + thrust * y, est["num"])
    den = xp.where(active, MASS_FORGET * est["den"] + y * y, est["den"])
    return {"num": num, "den": den, "thrust": thrust}


def mass_ratio(xp: Any, est: dict[str, Any], mass_nominal: float) -> Any:
    """Estimated mass relative to the nominal mass."""
    return xp.clip(est["num"] / est["den"] / mass_nominal, MASS_BOUNDS[0], MASS_BOUNDS[1])


def policy_observation(
    xp: Any, state: dict[str, Any], gate_sequence: Any, act_prev: Any, act_prev2: Any, ratio: Any
) -> Any:
    """Assemble the policy input for a batch of drones.

    Args:
        xp: The array namespace.
        state: Batched race observation with the keys of the environment's observation space.
        gate_sequence: The gate ID of each entry of the gate order (m,).
        act_prev: The last policy action (n, 3).
        act_prev2: The policy action before that (n, 3).
        ratio: The estimated relative mass (n,).

    Returns:
        The policy input (n, OBS_DIM).
    """
    pos, vel = state["pos"], state["vel"]
    n, n_passes = pos.shape[0], gate_sequence.shape[0]
    row = xp.arange(n)
    rot = quat_to_matrix(xp, state["quat"])

    # Target gate of the gate order and the one after it. Stays at the last one when finished.
    passed = state["n_gates_passed"]
    gate = gate_sequence[xp.minimum(passed, n_passes - 1)]
    next_gate = gate_sequence[xp.minimum(passed + 1, n_passes - 1)]

    gate_rot = quat_to_matrix(xp, state["gates_quat"][row, gate])
    gate_rel = state["gates_pos"][row, gate] - pos
    gate_rel_local = xp.sum(gate_rot * gate_rel[..., None], axis=-2)
    vel_local = xp.sum(gate_rot * vel[..., None], axis=-2)
    next_rot = quat_to_matrix(xp, state["gates_quat"][row, next_gate])
    next_rel = state["gates_pos"][row, next_gate] - pos
    obstacles_rel = state["obstacles_pos"][..., :2] - pos[:, None, :2]

    terms = [
        pos / POS_SCALE,
        vel / VEL_SCALE,
        rot[..., 0],
        rot[..., 2],
        state["ang_vel"] / ANG_VEL_SCALE,
        act_prev,
        act_prev2,
        gate_rel / POS_SCALE,
        gate_rel_local / POS_SCALE,
        vel_local / VEL_SCALE,
        gate_rot[..., 0],
        xp.float32(1.0) * state["gates_visited"][row, gate][:, None],
        next_rel / POS_SCALE,
        next_rot[..., 0],
        xp.float32(1.0) * state["gates_visited"][row, next_gate][:, None],
        obstacles_rel.reshape(n, -1) / POS_SCALE,
        xp.float32(1.0) * state["obstacles_visited"],
        xp.eye(n_passes + 1, dtype=xp.float32)[passed],
        ((ratio - 1.0) / MASS_RATIO_SCALE)[:, None],
    ]
    return xp.concatenate(terms, axis=-1)


def mlp_forward(xp: Any, layers: list[tuple[Any, Any]], x: Any) -> Any:
    """Forward pass of an ELU MLP with a tanh output, given its (weight, bias) layers."""
    for w, b in layers[:-1]:
        h = x @ w + b
        x = xp.where(h > 0, h, xp.exp(xp.minimum(h, 0.0)) - 1.0)
    w, b = layers[-1]
    return xp.tanh(x @ w + b)


class RLRacer(Controller):
    """Race controller running a trained policy."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load the policy.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        ckpt = np.load(POLICY_PATH)
        n_layers = int(ckpt["n_layers"])
        self._layers = [(ckpt[f"w{i}"], ckpt[f"b{i}"]) for i in range(n_layers)]
        self._limits = {k: float(ckpt[k]) for k in ("thrust_mid", "thrust_half", "max_tilt")}
        self._mass_nominal = float(ckpt["mass_nominal"])
        self._dt = 1.0 / config.env.freq
        self._gate_sequence = np.asarray(obs["gate_sequence"], dtype=int)
        self._n_passes = self._gate_sequence.shape[0]
        self.episode_reset()

    def episode_reset(self):
        """Reset the action history and the mass estimate."""
        self._act_prev = np.zeros((1, 3), dtype=np.float32)
        self._act_prev2 = np.zeros((1, 3), dtype=np.float32)
        self._est = mass_estimate_init(np, 1, self._mass_nominal)
        self._vel_prev = None
        self._thrust_cmd = np.full((1,), self._mass_nominal * G, dtype=np.float32)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next attitude command from the current observation.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information.

        Returns:
            The attitude command [roll, pitch, yaw, thrust].
        """
        state = {k: np.asarray(v)[None] for k, v in obs.items()}
        if self._vel_prev is not None:
            self._est = mass_estimate_update(
                np,
                self._est,
                state["pos"],
                state["quat"],
                state["vel"],
                self._vel_prev,
                self._thrust_cmd,
                self._limits,
                self._dt,
            )
        self._vel_prev = state["vel"]
        ratio = mass_ratio(np, self._est, self._mass_nominal)
        policy_in = policy_observation(
            np, state, self._gate_sequence, self._act_prev, self._act_prev2, ratio
        )
        action = mlp_forward(np, self._layers, policy_in.astype(np.float32))
        self._act_prev2, self._act_prev = self._act_prev, action
        command = action_to_command(np, action, self._limits)
        self._thrust_cmd = command[..., 3]
        return command[0].astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Finish the controller once all gates have been passed.

        Returns:
            True if the race is finished, False otherwise.
        """
        return int(obs["n_gates_passed"]) >= self._n_passes
