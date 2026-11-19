import crazyflow  # noqa: F401, must be imported before scipy
import jax.numpy as jp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.rl_racer import (
    action_to_command,
    mass_estimate_init,
    mass_estimate_update,
    mass_ratio,
    mlp_forward,
    policy_observation,
    quat_to_matrix,
)
from lsy_drone_racing.control.train_rl_racer import boxes_overlap

LIMITS = {"thrust_mid": 0.44, "thrust_half": 0.36, "max_tilt": 1.2}
MASS = 0.0434


def random_state(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "pos": rng.uniform(-2, 2, (n, 3)).astype(np.float32),
        "quat": R.random(n, random_state=seed).as_quat().astype(np.float32),
        "vel": rng.uniform(-3, 3, (n, 3)).astype(np.float32),
        "ang_vel": rng.uniform(-5, 5, (n, 3)).astype(np.float32),
        "n_gates_passed": rng.integers(0, 6, n),
        "gates_pos": rng.uniform(-2, 2, (n, 4, 3)).astype(np.float32),
        "gates_quat": R.random(4 * n, random_state=seed + 1).as_quat().reshape(n, 4, 4),
        "gates_visited": rng.integers(0, 2, (n, 4)).astype(bool),
        "obstacles_pos": rng.uniform(-2, 2, (n, 4, 3)).astype(np.float32),
        "obstacles_visited": rng.integers(0, 2, (n, 4)).astype(bool),
    }


@pytest.mark.unit
def test_quat_to_matrix():
    quat = R.random(16, random_state=0).as_quat()
    np.testing.assert_allclose(quat_to_matrix(np, quat), R.from_quat(quat).as_matrix(), atol=1e-12)


@pytest.mark.unit
def test_observation_numpy_matches_jax():
    """The controller (numpy) has to see exactly what the policy saw during training (JAX)."""
    n = 32
    state = random_state(n)
    sequence = np.array([0, 1, 2, 3, 1])
    act1, act2 = np.ones((n, 3), dtype=np.float32) * 0.5, np.zeros((n, 3), dtype=np.float32)
    ratio = np.linspace(0.9, 1.1, n, dtype=np.float32)
    obs_np = policy_observation(np, state, sequence, act1, act2, ratio)
    state_jp = {k: jp.asarray(v) for k, v in state.items()}
    obs_jp = policy_observation(
        jp, state_jp, jp.asarray(sequence), jp.asarray(act1), jp.asarray(act2), jp.asarray(ratio)
    )
    assert obs_np.shape == (n, 60)
    np.testing.assert_allclose(obs_np, np.asarray(obs_jp), atol=1e-5)


@pytest.mark.unit
def test_gate_frame_observation():
    """A drone behind a rotated gate is at a known position in the gate frame."""
    state = random_state(1)
    state["n_gates_passed"] = np.array([0])
    state["pos"] = np.zeros((1, 3), dtype=np.float32)
    state["gates_pos"][0, 0] = (1.0, 1.0, 0.0)
    state["gates_quat"][0, 0] = R.from_euler("z", np.pi / 4).as_quat()
    obs = policy_observation(
        np, state, np.array([0, 1, 2, 3, 1]), np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1)
    )
    # The local gate position follows pos, vel, x axis, z axis, ang vel and the two actions
    local = obs[0, 21 + 3 : 21 + 6] * 2.0
    np.testing.assert_allclose(local, [np.sqrt(2), 0.0, 0.0], atol=1e-5)


@pytest.mark.unit
def test_action_to_command():
    command = action_to_command(np, np.array([[2.0, -2.0, 1.0], [0.0, 0.5, -1.0]]), LIMITS)
    np.testing.assert_allclose(command[0], [1.2, -1.2, 0.0, 0.8])
    np.testing.assert_allclose(command[1], [0.0, 0.6, 0.0, 0.08])


@pytest.mark.unit
@pytest.mark.parametrize("true_ratio", [0.9, 1.0, 1.1])
def test_mass_estimate_converges(true_ratio: float):
    """Level flight at the thrust that holds the true mass reveals the mass."""
    dt = 0.02
    quat, pos = np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 1.0]])
    thrust = 0.5
    acc = thrust / (MASS * true_ratio) - 9.81
    est = mass_estimate_init(np, 1, MASS)
    vel = np.zeros((1, 3))
    for _ in range(200):
        vel_next = vel + np.array([[0.0, 0.0, acc * dt]])
        cmd = np.array([thrust])
        est = mass_estimate_update(np, est, pos, quat, vel_next, vel, cmd, LIMITS, dt)
        vel = vel_next
    assert mass_ratio(np, est, MASS)[0] == pytest.approx(true_ratio, abs=0.01)


@pytest.mark.unit
def test_mlp_forward_elu_tanh():
    layers = [(np.eye(2), np.zeros(2)), (np.eye(2), np.zeros(2))]
    out = mlp_forward(np, layers, np.array([[1.0, -1.0]]))
    np.testing.assert_allclose(out, np.tanh([[1.0, np.exp(-1.0) - 1.0]]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("offset", "yaw", "overlap"),
    [
        ((0.0, 0.0, 0.0), 0.0, True),
        ((1.9, 0.0, 0.0), 0.0, True),  # Faces touch nearly, half sizes add up to 2
        ((2.1, 0.0, 0.0), 0.0, False),
        ((2.3, 0.0, 0.0), np.pi / 4, True),  # Only the rotated corner reaches the other box
        ((2.5, 0.0, 0.0), np.pi / 4, False),
        ((1.0, 1.0, 2.1), 0.0, False),
    ],
)
def test_boxes_overlap(offset: tuple[float, float, float], yaw: float, overlap: bool):
    half = jp.ones(3)
    rot = jp.asarray(R.from_euler("z", yaw).as_matrix(), dtype=jp.float32)
    result = boxes_overlap(jp.zeros(3), jp.eye(3), half, jp.asarray(offset), rot, half)
    assert bool(result) == overlap
