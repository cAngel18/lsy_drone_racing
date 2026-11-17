from pathlib import Path

import gymnasium
import numpy as np
import pytest
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller


@pytest.mark.integration
def test_mppi_racer_flies_a_random_track():
    """The MPPI controller plans and flies through all gates of a random level 3 track."""
    root = Path(__file__).parents[2]
    config = load_config(root / "config/level3.toml")
    config.sim.render = False
    ctrl_cls = load_controller(root / "lsy_drone_racing/control/mppi_racer.py")
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode="attitude",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=1337,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset(seed=1337)
    ctrl = ctrl_cls(obs, info, config)
    assert env.action_space.contains(ctrl.compute_control(obs, info).astype(np.float32))
    for _ in range(1500):
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if ctrl.step_callback(action, obs, reward, terminated, truncated, info):
            break
        if terminated or truncated:
            break
    env.close()
    assert obs["n_gates_passed"] == obs["gate_sequence"].shape[0]
