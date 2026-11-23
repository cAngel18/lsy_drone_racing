"""MPPI racing controller: glue between the env, the racing problem and the generic MPPI engine.

* ``mppi_engine.py``: The MPPI algorithm itself (sampling, rollouts, weighting, warm start).
* ``racing_problem.py``: Drone model and racing cost.
* This file: Builds the engine once, converts every observation into the rollout start state and
  the problem parameters, and estimates an acceleration bias to compensate for model mismatch.

Usage:
    Set ``control_mode = "attitude"`` in the ``[env]`` section of your level config, then
    ``python scripts/sim.py --config level2.toml --controller christians_mppi/mppi_controller.py``
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim.visualize import draw_line

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.christians_mppi import racing_problem as rp
from lsy_drone_racing.control.christians_mppi.mppi_engine import MPPI, MPPIConfig

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# MPPI hyperparameters. The input bounds are added from the drone model in __init__.
MPPI_SETTINGS = dict(
    n_samples=1024,
    horizon=30,  # 0.6 s lookahead at 50 Hz
    temperature=2.0,
    noise_std=(0.2, 0.2, 0.0, 0.08),  # roll, pitch, yaw, thrust
    noise_corr=0.8,
)
RACING_SETTINGS = dict()  # Overrides of the RacingConfig defaults (cost weights, ...)

# Disturbance observer
BIAS_GAIN = 0.1  # Low-pass gain of the acceleration bias estimate
BIAS_LIMIT = 3.0  # [m/s^2]


class MPPIController(Controller):
    """MPPI controller that commands collective thrust and attitude."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Set up the racing problem and the MPPI engine, and compile the MPPI step."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError('MPPIRacer needs `control_mode = "attitude"` in [env].')

        self.racing_cfg = rp.RacingConfig(dt=1.0 / config.env.freq, **RACING_SETTINGS)
        self.model = rp.load_model_params(config.sim.drone)
        self.hover = np.array([0.0, 0.0, 0.0, rp.hover_thrust(self.model)], dtype=np.float32)
        u_min, u_max = rp.input_bounds(self.model, self.racing_cfg)

        self.engine = MPPI(
            MPPIConfig(u_min=u_min, u_max=u_max, **MPPI_SETTINGS),
            dynamics=partial(rp.dynamics, cfg=self.racing_cfg),
            running_cost=partial(rp.running_cost, cfg=self.racing_cfg),
        )
        self._predict = jax.jit(partial(rp.so_rpy_step, dt=self.racing_cfg.dt))

        limits = config.env.track.safety_limits
        self.pos_low = np.asarray(limits.pos_limit_low, dtype=np.float32)
        self.pos_high = np.asarray(limits.pos_limit_high, dtype=np.float32)
        seed = config.env.seed if config.env.seed >= 0 else np.random.randint(2**31)
        self._seed = int(seed)
        self.episode_reset()

        # Compile now instead of in the first control step of the race, then start fresh
        self.compute_control(obs, info)
        self._predict(self._drone_prev, self.hover, self.model, self._acc_bias)
        self.episode_reset()

    def episode_reset(self):
        """Reset the nominal plan, random key and disturbance estimate."""
        self.mppi_state = self.engine.init(self.hover, seed=self._seed)
        self._u_last = self.hover.copy()
        self._acc_bias = np.zeros(3, dtype=np.float32)
        self._drone_prev = None
        self.info = None  # MPPIInfo of the last step, for debugging and rendering

    def reset(self):
        """Reset internal variables."""
        self.episode_reset()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Run one MPPI iteration and return the attitude command [roll, pitch, yaw, thrust]."""
        drone = rp.obs_to_drone_state(obs)
        self._update_disturbance(obs)

        x0 = rp.initial_state(drone, self._u_last, int(obs["n_gates_passed"]))
        params = rp.RacingParams(
            model=self.model,
            track=rp.build_track(obs, self.pos_low, self.pos_high, self.racing_cfg),
            acc_bias=jnp.asarray(self._acc_bias),
        )
        self.mppi_state, u, self.info = self.engine.step(self.mppi_state, x0, params)

        action = np.asarray(u, dtype=np.float32)
        self._drone_prev, self._u_last = drone, action
        return action

    def _update_disturbance(self, obs: dict[str, NDArray[np.floating]]):
        """Compare the predicted and measured velocity and low-pass filter the difference."""
        if self._drone_prev is None or obs["pos"][2] < 0.05:  # On the ground, contacts dominate
            return
        predicted = self._predict(
            self._drone_prev, jnp.asarray(self._u_last), self.model, jnp.asarray(self._acc_bias)
        )
        acc_error = (obs["vel"] - np.asarray(predicted[6:9])) / self.racing_cfg.dt
        bias = self._acc_bias + BIAS_GAIN * acc_error
        self._acc_bias = np.clip(bias, -BIAS_LIMIT, BIAS_LIMIT).astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Never finish the episode from the controller side."""
        return False

    def render_callback(self, sim: Sim):
        """Draw the path that the current MPPI plan is predicted to fly."""
        if self.info is not None:
            path = np.asarray(self.info.trajectory.drone[:, :3])
            draw_line(sim, path, rgba=np.array([0.0, 1.0, 0.0, 1.0]))
