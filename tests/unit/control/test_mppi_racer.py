import crazyflow  # noqa: F401, must be imported before scipy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from crazyflow.dynamics import Dynamics, load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.mppi_racer import (
    DRONE_RADIUS,
    HORIZON,
    MAX_TILT,
    NOISE_SMOOTHING,
    NOISE_STD,
    PATH_DS,
    POLE_CLEARANCE,
    POLE_RADIUS,
    SPEED_MAX,
    G,
    ThrustGainEstimator,
    body_z,
    gate_axes,
    model_step,
    mppi_step,
    plan_path,
    sample_noise,
    speed_profile,
)
from lsy_drone_racing.envs.randomize import build_random_track_fn

PARAMS = load_params(Dynamics.so_rpy, "cf21B_500")
COEF = {
    k: jnp.asarray(PARAMS[k], dtype=jnp.float32)
    for k in ("rpy_coef", "rpy_rates_coef", "cmd_rpy_coef")
}
GAIN = float(PARAMS["cmd_f_coef"]) / float(PARAMS["mass"])
THRUST_RANGE = (4 * float(PARAMS["thrust_min"]), 4 * float(PARAMS["thrust_max"]))
BOUNDS = (
    jnp.asarray([-MAX_TILT, -MAX_TILT, THRUST_RANGE[0]]),
    jnp.asarray([MAX_TILT, MAX_TILT, THRUST_RANGE[1]]),
)
ARENA = (np.array([-2.5, -1.5, -1e-3]), np.array([2.5, 1.5, 2.0]))
SEQUENCE = np.array([0, 1, 2, 3, 1])
DT = 0.02


def random_tracks(n: int) -> list[tuple[np.ndarray, ...]]:
    """Tracks from the generator of the environment, as in level 3."""
    sample = build_random_track_fn(
        np.array([0.7, 1.2, 0.7, 1.2]), np.full(4, 1.55), ARENA[0][:2], ARENA[1][:2]
    )
    tracks = []
    for i in range(n):
        start = np.array([0.0, 0.0, 0.01])
        gates_pos, gates_quat, poles = sample(jnp.asarray(start), jax.random.key(i))
        tracks.append(
            (start, *(np.asarray(x, dtype=np.float64) for x in (gates_pos, gates_quat, poles)))
        )
    return tracks


def scene(gates_pos: np.ndarray, gates_quat: np.ndarray, poles: np.ndarray) -> dict:
    return {
        "gates_pos": jnp.asarray(gates_pos, dtype=jnp.float32),
        "gates_rot": jnp.asarray(R.from_quat(gates_quat).as_matrix(), dtype=jnp.float32),
        "poles": jnp.asarray(poles, dtype=jnp.float32),
        "low": jnp.asarray(ARENA[0] + 0.1, dtype=jnp.float32),
        "high": jnp.asarray(ARENA[1] - 0.1, dtype=jnp.float32),
    }


def straight_reference(start: np.ndarray, speed: float) -> tuple:
    t = DT * np.arange(1, HORIZON + 1)[:, None]
    direction = np.array([1.0, 0.0, 0.0])
    pos = start + speed * t * direction
    tangent = np.tile(direction, (HORIZON, 1))
    return tuple(jnp.asarray(x, dtype=jnp.float32) for x in (pos, tangent, speed * tangent))


def run_mppi(state0: np.ndarray, refs: tuple, objects: dict, iterations: int) -> dict:
    """Iterate MPPI from a fixed state, warm starting from its own solution."""
    hover = G / GAIN
    u = jnp.tile(jnp.asarray([0.0, 0.0, hover]), (HORIZON, 1))
    for i in range(iterations):
        out = mppi_step(
            jax.random.key(i),
            jnp.asarray(state0, dtype=jnp.float32),
            u,
            u[0],
            refs,
            objects,
            jnp.float32(GAIN),
            BOUNDS,
            COEF,
            dt=DT,
        )
        u = out["u"]
    return out


@pytest.mark.unit
def test_body_z_matches_scipy():
    rpy = np.random.default_rng(0).uniform(-1.0, 1.0, (32, 3))
    expected = R.from_euler("xyz", rpy).as_matrix()[..., 2]
    np.testing.assert_allclose(body_z(jnp.asarray(rpy)), expected, atol=1e-6)


@pytest.mark.unit
def test_model_hovers_and_follows_tilt():
    """Hover thrust keeps the drone in place, a pitch command accelerates it along +x."""
    state = jnp.zeros(12).at[2].set(1.0)
    hover = jnp.asarray([0.0, 0.0, G / GAIN])
    hovering = state
    for _ in range(50):
        hovering = model_step(hovering, hover, GAIN, COEF, DT)
    np.testing.assert_allclose(hovering, state, atol=1e-5)

    pitched = state
    for _ in range(50):
        pitched = model_step(pitched, jnp.asarray([0.0, 0.3, 1.1 * G / GAIN]), GAIN, COEF, DT)
    assert pitched[6] > 1.0  # Velocity along +x
    steady = PARAMS["cmd_rpy_coef"][1] / -PARAMS["rpy_coef"][1] * 0.3
    assert float(pitched[4]) == pytest.approx(steady, abs=0.01)


@pytest.mark.unit
def test_noise_is_smooth_with_the_configured_spread():
    noise = np.asarray(sample_noise(jax.random.key(0), 4096, HORIZON))
    assert noise.shape == (4096, HORIZON, 3)
    np.testing.assert_allclose(noise.std(axis=(0, 1)), NOISE_STD, rtol=0.05)
    lag1 = np.mean(noise[:, 1:] * noise[:, :-1], axis=(0, 1)) / noise.var(axis=(0, 1))
    np.testing.assert_allclose(lag1, NOISE_SMOOTHING, atol=0.05)


@pytest.mark.unit
@pytest.mark.parametrize("track", random_tracks(20))
def test_path_passes_the_gates_and_keeps_clear(track: tuple):
    start, gates_pos, gates_quat, poles = track
    path, gate_index = plan_path(start, gates_pos, gates_quat, poles, SEQUENCE, ARENA, {})
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.median(steps) == pytest.approx(PATH_DS, abs=1e-3)
    assert steps.max() <= PATH_DS + 1e-6
    normal, _ = gate_axes(gates_quat)
    for k, gate in enumerate(SEQUENCE):
        assert np.linalg.norm(path[gate_index[k]] - gates_pos[gate]) < PATH_DS
        direction = path[gate_index[k] + 2] - path[gate_index[k] - 2]
        assert direction @ normal[gate] / np.linalg.norm(direction) > 0.95
    # POLE_CLEARANCE is the aim. Narrow passages between objects can leave less, but never so
    # little that the drone would touch a pole at its nominal position.
    distance = np.linalg.norm(path[:, None, :2] - poles[None, :, :2], axis=-1)
    assert distance.min() > POLE_RADIUS + DRONE_RADIUS + 0.03
    assert np.mean(distance.min(axis=1) > 0.9 * POLE_CLEARANCE) > 0.95
    # The path stays in the arena up to the last gate, the extension after it may leave
    inside = path[: gate_index[-1] + 1]
    assert np.all(inside >= ARENA[0]) and np.all(inside <= ARENA[1])


def piece_sides(path: np.ndarray, gate_index: np.ndarray, poles: np.ndarray) -> dict:
    """Side of each nearby pole relative to each piece of a path between two gates."""
    bounds = np.concatenate([[0], gate_index])
    sides = {}
    for k in range(len(gate_index)):
        part = path[bounds[k] : bounds[k + 1] + 1]
        for j, pole in enumerate(poles):
            distance = np.linalg.norm(part[:, :2] - pole[:2], axis=-1)
            i = int(np.argmin(distance))
            if distance[i] > 0.6 or i == 0 or i == len(part) - 1:
                continue
            tangent, rel = part[i + 1] - part[i - 1], pole[:2] - part[i, :2]
            sides[(k, j)] = np.sign(tangent[0] * rel[1] - tangent[1] * rel[0])
    return sides


@pytest.mark.unit
def test_replanning_mostly_keeps_the_side_of_shifted_poles():
    """A pole that turns out to be elsewhere is usually still passed on the side that was planned.

    The planner remembers the side of every detour. A pole that moves out of the way of a straight
    piece needs no detour any more, though, and then the path can end up on its other side. In
    that case MPPI still keeps the clearance to the observed pole.
    """
    rng = np.random.default_rng(0)
    kept = []
    for start, gates_pos, gates_quat, poles in random_tracks(40):
        sides = {}
        path, gate_index = plan_path(start, gates_pos, gates_quat, poles, SEQUENCE, ARENA, sides)
        shifted = poles + np.c_[rng.uniform(-0.15, 0.15, (len(poles), 2)), np.zeros(len(poles))]
        new_path, new_index = plan_path(
            start, gates_pos, gates_quat, shifted, SEQUENCE, ARENA, sides
        )
        before = piece_sides(path, gate_index, poles)
        after = piece_sides(new_path, new_index, shifted)
        kept += [before[key] == after[key] for key in before.keys() & after.keys()]
    assert len(kept) > 100
    assert np.mean(kept) > 0.9


@pytest.mark.unit
def test_speed_profile_respects_limits():
    straight = np.c_[np.arange(0.0, 6.0, PATH_DS), np.zeros(120), np.ones(120)]
    angle = np.arange(0.0, 2 * np.pi, PATH_DS / 0.3)
    circle = np.c_[0.3 * np.cos(angle), 0.3 * np.sin(angle), np.ones_like(angle)]
    for path in (straight, circle):
        speed, time = speed_profile(path)
        assert speed[0] == 0.0
        assert np.all(speed <= SPEED_MAX + 1e-9)
        assert np.all(np.diff(time) > 0.0)
    assert speed_profile(straight)[0].max() == pytest.approx(SPEED_MAX)
    assert speed_profile(circle)[0].max() < 0.7 * SPEED_MAX


@pytest.mark.unit
def test_mppi_follows_the_reference():
    state0 = np.zeros(12)
    state0[2] = 1.0
    objects = scene(
        np.array([[0.0, 2.0, 0.7]]), np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[0.0, -2.0, 1.55]])
    )
    out = run_mppi(state0, straight_reference(state0[:3], 1.0), objects, 10)
    u = np.asarray(out["u"])
    assert u[0, 1] > 0.05  # Pitch forward, towards +x
    assert np.all(u >= np.asarray(BOUNDS[0]) - 1e-6) and np.all(u <= np.asarray(BOUNDS[1]) + 1e-6)
    assert np.asarray(out["path"])[-1, 0] > 0.2


@pytest.mark.unit
def test_mppi_flies_around_a_pole_on_the_reference():
    """The reference leads straight through a pole, the predicted trajectory passes beside it."""
    state0 = np.zeros(12)
    state0[2], state0[6] = 1.0, 1.5
    pole = np.array([[0.6, 0.0, 1.55]])
    objects = scene(np.array([[0.0, 3.0, 0.7]]), np.array([[0.0, 0.0, 0.0, 1.0]]), pole)
    out = run_mppi(state0, straight_reference(state0[:3], 1.5), objects, 15)
    path = np.asarray(out["path"])
    distance = np.linalg.norm(path[:, :2] - pole[0, :2], axis=-1)
    assert distance.min() > DRONE_RADIUS + POLE_RADIUS


@pytest.mark.unit
@pytest.mark.parametrize("true_ratio", [0.9, 1.0, 1.1])
def test_thrust_gain_estimate_converges(true_ratio: float):
    """Climbing at a constant thrust reveals the gain of the thrust command."""
    estimator = ThrustGainEstimator(GAIN, G / GAIN, THRUST_RANGE)
    quat, pos = np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
    thrust = 0.5
    acc = np.array([0.0, 0.0, true_ratio * GAIN * thrust - G])
    for _ in range(200):
        estimator.update(pos, quat, acc, thrust)
    assert estimator.gain / GAIN == pytest.approx(true_ratio, abs=0.01)
