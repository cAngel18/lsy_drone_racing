from typing import Any

import crazyflow  # noqa: F401, must be imported before scipy
import jax
import jax.numpy as jp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.rl_general import (
    LINE_AHEAD,
    LINE_DS,
    LINE_LENGTH,
    OBS_DIM,
    along_and_across,
    closest_solids,
    collision_course,
    gate_frame,
    general_command,
    general_observation,
    line_ahead,
    plan_line,
    project_on_line,
    repair_line,
    to_frame,
)
from lsy_drone_racing.control.train_rl_general import add_pole_hazard, line_tables

LIMITS = {"thrust_mid": 0.44, "thrust_half": 0.36, "max_tilt": 1.4}
SEQUENCE = np.array([0, 1, 2, 3, 1])
ARENA = (np.array([-2.5, -1.5]), np.array([2.5, 1.5]))


def random_state(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    gates_yaw = rng.uniform(-3, 3, (n, 4))
    quat_tilt = rng.uniform(-0.3, 0.3, (n, 2))
    return {
        "pos": rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        "quat": R.from_euler("xyz", np.c_[quat_tilt, np.zeros(n)]).as_quat().astype(np.float32),
        "vel": rng.uniform(-3, 3, (n, 3)).astype(np.float32),
        "ang_vel": rng.uniform(-5, 5, (n, 3)).astype(np.float32),
        "n_gates_passed": rng.integers(0, 6, n),
        "gates_pos": rng.uniform(-2, 2, (n, 4, 3)).astype(np.float32),
        "gates_quat": R.from_euler("z", gates_yaw.reshape(-1, 1)).as_quat().reshape(n, 4, 4),
        "gates_visited": rng.integers(0, 2, (n, 4)).astype(bool),
        "obstacles_pos": rng.uniform(-2, 2, (n, 4, 3)).astype(np.float32),
        "obstacles_visited": rng.integers(0, 2, (n, 4)).astype(bool),
    }


def observation(xp: Any, state: dict, act: np.ndarray, arena: tuple = ARENA) -> np.ndarray:
    n = state["pos"].shape[0]
    # Points ahead rotate with the scene, because they are relative to the drone in the gate frame
    ahead = state["pos"][:, None] + xp.ones((n, len(LINE_AHEAD), 3))
    if "ahead" in state:
        ahead = state["ahead"]
    state = {k: v for k, v in state.items() if k != "ahead"}
    return general_observation(
        xp,
        state,
        xp.asarray(SEQUENCE),
        xp.asarray(act),
        xp.asarray(act),
        xp.ones(n),
        xp.full(n, 3.0),
        (xp.asarray(arena[0]), xp.asarray(arena[1])),
        xp.asarray(ahead),
    )


@pytest.mark.unit
def test_observation_numpy_matches_jax():
    """The controller (numpy) has to see exactly what the policy saw during training (JAX)."""
    n = 32
    state = random_state(n)
    act = np.full((n, 3), 0.5, dtype=np.float32)
    obs_np = observation(np, state, act)
    obs_jp = observation(jp, {k: jp.asarray(v) for k, v in state.items()}, act)
    assert obs_np.shape == (n, OBS_DIM)
    np.testing.assert_allclose(obs_np, np.asarray(obs_jp), atol=1e-4)


@pytest.mark.unit
def test_observation_is_invariant_to_rotating_the_scene():
    """A track and its yaw rotation have to look identical to the policy."""
    n = 16
    state = random_state(n)
    act = np.random.default_rng(1).uniform(-1, 1, (n, 3))
    # No arena: its limits are the one thing that is fixed in the world
    arena = (np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    yaw = 1.1
    rotation = R.from_euler("z", yaw)
    state["ahead"] = state["pos"][:, None] + np.random.default_rng(2).uniform(-1, 1, (n, 5, 3))
    rotated = dict(state)
    for key in ("pos", "vel"):
        rotated[key] = rotation.apply(state[key])
    rotated["ahead"] = rotation.apply(state["ahead"].reshape(-1, 3)).reshape(state["ahead"].shape)
    for key in ("gates_pos", "obstacles_pos"):
        rotated[key] = rotation.apply(state[key].reshape(-1, 3)).reshape(state[key].shape)
    rotated["quat"] = (rotation * R.from_quat(state["quat"])).as_quat()
    gates = rotation * R.from_quat(state["gates_quat"].reshape(-1, 4))
    rotated["gates_quat"] = gates.as_quat().reshape(state["gates_quat"].shape)
    act_rotated = act.copy()
    act_rotated[:, :2] = rotation.apply(np.c_[act[:, :2], np.zeros(n)])[:, :2]
    original = observation(np, state, act, arena)
    np.testing.assert_allclose(original, observation(np, rotated, act_rotated, arena), atol=1e-5)


@pytest.mark.unit
def test_to_frame_inverts_the_frame_rotation():
    state = random_state(8)
    c, s = gate_frame(np, state, SEQUENCE)
    row = np.arange(8)
    target = SEQUENCE[np.minimum(state["n_gates_passed"], 4)]
    normal = R.from_quat(state["gates_quat"][row, target]).apply([1.0, 0.0, 0.0])
    local = to_frame(np, normal, c, s)
    np.testing.assert_allclose(local, np.tile([1.0, 0.0, 0.0], (8, 1)), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("frame_yaw", [0.0, 0.7, -2.5])
@pytest.mark.parametrize("action", [(0.5, 0.0), (0.0, 0.5), (-0.3, 0.4), (1.0, 1.0)])
def test_command_tilts_the_thrust_axis_as_requested(frame_yaw: float, action: tuple[float, float]):
    """The commanded attitude has its thrust axis tilted towards the action in the gate frame."""
    c, s = np.array([np.cos(frame_yaw)]), np.array([np.sin(frame_yaw)])
    command, world = general_command(np, np.array([[*action, 0.0]]), c, s, LIMITS)
    z = R.from_euler("xyz", [command[0, 0], command[0, 1], 0.0]).apply([0.0, 0.0, 1.0])
    size = min(np.hypot(*action), 1.0)
    tilt = size * LIMITS["max_tilt"]
    assert np.arccos(z[2]) == pytest.approx(tilt, abs=1e-6)
    direction = np.array(action) / np.hypot(*action)
    world_direction = np.array(
        [c[0] * direction[0] - s[0] * direction[1], s[0] * direction[0] + c[0] * direction[1]]
    )
    np.testing.assert_allclose(z[:2] / np.linalg.norm(z[:2]), world_direction, atol=1e-6)
    np.testing.assert_allclose(world[0, :2] / size, world_direction, atol=1e-6)


def track_arrays(yaws: tuple, poles: list | None = None) -> tuple:
    gates = np.array(
        [[1.0, 0.0, 0.7], [2.0, 0.5, 1.2], [1.0, 1.0, 0.7], [0.0, 1.0, 1.2]], dtype=np.float32
    )
    quats = R.from_euler("z", np.array(yaws)[:, None]).as_quat().astype(np.float32)
    far = [[3.0, 3.0, 1.5]] * 4
    obstacles = np.array((poles or []) + far, dtype=np.float32)[:4]
    return np.zeros(3, dtype=np.float32), gates, quats, obstacles


@pytest.mark.unit
def test_line_enters_gates_along_their_normals():
    """The line runs through each gate, also when a gate is entered from behind."""
    # Gate 1 faces away from gate 3, so the last segment has to loop around it
    yaws = (0.0, np.pi, np.pi, 0.0)
    start, gates, quats, poles = track_arrays(yaws)
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    assert line.shape == (LINE_LENGTH, 3)
    assert np.all(np.diff(index) > 0)
    for k, gate in enumerate(SEQUENCE):
        np.testing.assert_allclose(line[index[k]], gates[gate], atol=LINE_DS)
        direction = line[index[k] + 1] - line[index[k] - 1]
        normal = np.array([np.cos(yaws[gate]), np.sin(yaws[gate])])
        cosine = direction[:2] @ normal / np.linalg.norm(direction[:2])
        assert cosine > 0.9, f"pass {k} does not follow the gate normal"


@pytest.mark.unit
def test_line_has_constant_sample_distance():
    start, gates, quats, poles = track_arrays((0.0, 1.5, 3.0, 0.0))
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    steps = np.linalg.norm(np.diff(line[: index[-1]], axis=0), axis=-1)
    np.testing.assert_allclose(steps, LINE_DS, atol=0.01)


@pytest.mark.unit
def test_line_numpy_matches_jax():
    """The controller has to plan the line that the policy was trained on."""
    arrays = track_arrays((0.0, 1.5, 3.0, 0.0), [[0.5, -0.1, 1.5]])
    line_np, index_np = plan_line(np, *arrays, SEQUENCE)
    line_jp, index_jp = plan_line(jp, *(jp.asarray(x) for x in arrays), jp.asarray(SEQUENCE))
    np.testing.assert_allclose(line_np, np.asarray(line_jp), atol=0.01)
    np.testing.assert_array_equal(index_np, np.asarray(index_jp))


@pytest.mark.unit
def test_line_goes_around_a_pole_in_its_way():
    """A pole halfway on the straight line between two gates leads to a detour."""
    pole = [1.5, 0.0, 1.5]
    gates = np.array(
        [[0.5, 0.0, 0.7], [2.5, 0.0, 1.2], [2.5, 1.2, 0.7], [0.5, 1.2, 1.2]], dtype=np.float32
    )
    quats = R.from_euler("z", np.array([0.0, 0.0, np.pi, np.pi])[:, None]).as_quat()
    _, _, _, poles = track_arrays((0.0, 0.0, 0.0, 0.0), [pole])
    start = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    line, index = plan_line(np, start, gates, quats.astype(np.float32), poles, SEQUENCE)
    distance = np.linalg.norm(line[index[0] : index[1], :2] - np.array(pole[:2]), axis=-1)
    assert distance.min() > 0.2


@pytest.mark.unit
def test_line_goes_around_the_gate_it_just_passed():
    """With the next gate straight behind, the line must not run back through the stand."""
    gates = np.array(
        [[1.0, 0.0, 1.2], [-0.5, 0.0, 0.7], [-0.5, 1.0, 1.2], [1.0, 1.0, 0.7]], dtype=np.float32
    )
    quats = R.from_euler("z", np.array([0.0, np.pi, 0.0, 0.0])[:, None]).as_quat()
    quats = quats.astype(np.float32)
    start, _, _, poles = track_arrays((0.0, 0.0, 0.0, 0.0))
    start = np.array([-0.5, -1.0, 0.0], dtype=np.float32)
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    piece = line[index[0] + 6 : index[1] - 6]
    to_gate = closest_solids(np, piece, gates[:1], quats[:1])
    assert np.linalg.norm(to_gate, axis=-1).min() > 0.15


@pytest.mark.unit
def test_projection_follows_the_line_and_stops_at_the_next_gate():
    start, gates, quats, poles = track_arrays((0.0, 1.5, 3.0, 0.0))
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    lines, gate_index, track = line[None], index[None], np.zeros(1, dtype=int)
    sample = np.array([index[0] - 4])
    idx, progress, distance = project_on_line(
        np, lines, gate_index, track, sample, line[sample] + 0.01, np.array([0])
    )
    assert abs(int(idx[0]) - int(sample[0])) <= 1
    assert progress[0] == pytest.approx(sample[0] * LINE_DS, abs=2 * LINE_DS)
    assert distance[0] < 0.05
    # A drone beyond the first gate makes no progress beyond it before that gate has been passed
    beyond = line[index[0] + 5][None]
    idx, progress, _ = project_on_line(
        np, lines, gate_index, track, np.array([index[0]]), beyond, np.array([0])
    )
    assert int(idx[0]) == int(index[0])
    idx, _, _ = project_on_line(
        np, lines, gate_index, track, np.array([index[0]]), beyond, np.array([1])
    )
    assert int(idx[0]) > int(index[0])
    assert line_ahead(np, lines, track, idx).shape == (1, len(LINE_AHEAD), 3)


@pytest.mark.unit
def test_reference_states_keep_clear_of_objects():
    pole = [0.5, 0.3, 1.5]
    start, gates, quats, poles = track_arrays((0.0, 1.5, 3.0, 0.0), [pole])
    arrays = (jp.asarray(x) for x in (start, gates, quats, poles))
    line, index = plan_line(jp, *arrays, jp.asarray(SEQUENCE))
    tables = line_tables(line, index, jp.asarray(gates), jp.asarray(poles), 0.7)
    valid = np.asarray(tables["valid"])
    assert valid.any()
    line = np.asarray(line)
    assert np.min(np.linalg.norm(line[valid, None] - gates[None], axis=-1)) > 0.5
    assert np.min(np.linalg.norm(line[valid, :2] - np.array(pole[:2]), axis=-1)) > 0.25
    assert np.min(line[valid, 2]) > 0.15
    # The first gate comes into sensor range before it is reached
    assert int(tables["gates_seen"][0]) < int(index[0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("local", "expected"),
    [
        ((0.0, 0.0, 0.05), (0.0, 0.0, 0.15)),  # In the opening, closer to the top bar
        ((0.0, 0.1, 0.0), (0.0, 0.1, 0.0)),  # Towards the side bar at y = 0.2
        ((0.5, 0.05, 0.0), (-0.49, 0.15, 0.0)),  # In front of the opening, closer to a side bar
        ((0.0, 0.0, -0.6), (0.0, 0.0, 0.0)),  # Inside the stand, which reaches up to -0.36
        ((0.0, 0.0, 0.5), (0.0, 0.0, -0.14)),  # Above the frame, whose top is at 0.36
    ],
)
def test_closest_solid_matches_gate_geometry(local: tuple, expected: tuple):
    """The distance to the frame and stand is measured in the frame of the gate."""
    yaw = 0.7
    rotation = R.from_euler("z", yaw)
    gate = np.array([[[1.0, -0.5, 1.2]]])
    quat = rotation.as_quat()[None, None]
    pos = gate[0] + rotation.apply(np.array(local))[None]
    vector = closest_solids(np, pos, gate, quat)[0, 0]
    np.testing.assert_allclose(rotation.inv().apply(vector), expected, atol=1e-6)


@pytest.mark.unit
def test_along_and_across_direction_of_travel():
    direction = np.array([[0.0, 1.0]])
    vectors = np.array([[[0.0, 2.0], [1.0, 0.0]]])
    result = along_and_across(np, vectors, direction)
    np.testing.assert_allclose(result[0], [[2.0, 0.0], [0.0, 1.0]])


@pytest.mark.unit
def test_collision_course_shows_miss_distance_and_time():
    velocity = np.array([[2.0, 0.0]])
    # Dead ahead at 1 m, 10 cm to the side at 1 m, far to the side, and behind
    vectors = np.array([[[1.0, 0.0], [1.0, 0.1], [1.0, 5.0], [-1.0, 0.0]]])
    course = collision_course(np, vectors, velocity)[0]
    assert course[0, 0] == pytest.approx(0.0)
    assert abs(course[1, 0]) == pytest.approx(0.25, abs=0.01)
    assert abs(course[2, 0]) == pytest.approx(1.0)
    assert course[0, 1] > 0.0
    assert course[0, 1] == pytest.approx(course[1, 1], abs=1e-6)
    assert course[3, 1] < 0.0


@pytest.mark.unit
def test_repair_keeps_a_line_without_reveals():
    start, gates, quats, poles = track_arrays((0.0, 1.5, 3.0, 0.0))
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    repaired = repair_line(np, line, index, SEQUENCE, gates, quats, gates, quats, poles)
    assert np.max(np.linalg.norm(repaired - line, axis=-1)) < 0.05


@pytest.mark.unit
def test_repair_follows_a_revealed_gate():
    """The line passes through the true gate, along its true normal."""
    start, gates, quats, poles = track_arrays((0.0, 1.5, 3.0, 0.0))
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    moved = gates.copy()
    moved[0] += np.array([0.1, -0.15, 0.08], dtype=np.float32)
    turned = quats.copy()
    turned[0] = R.from_euler("z", 0.2).as_quat()
    repaired = repair_line(np, line, index, SEQUENCE, gates, quats, moved, turned, poles)
    np.testing.assert_allclose(repaired[index[0]], moved[0], atol=LINE_DS)
    direction = repaired[index[0] + 2] - repaired[index[0] - 2]
    cosine = direction[:2] @ np.array([np.cos(0.2), np.sin(0.2)]) / np.linalg.norm(direction[:2])
    assert cosine > 0.99
    # The other passes do not move
    assert np.linalg.norm(repaired[index[2]] - line[index[2]]) < 0.01


@pytest.mark.unit
def test_repair_moves_away_from_a_revealed_pole():
    start, gates, quats, poles = track_arrays((0.0, 1.5, 3.0, 0.0))
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    # A pole that turns out to be right on the line, between two gates
    sample = (index[1] + index[2]) // 2
    revealed = poles.copy()
    revealed[0, :2] = line[sample, :2] + 0.02
    repaired = repair_line(np, line, index, SEQUENCE, gates, quats, gates, quats, revealed)
    piece = repaired[sample - 12 : sample + 12, :2]
    assert np.linalg.norm(piece - revealed[0, :2], axis=-1).min() > 0.15


@pytest.mark.unit
def test_repair_numpy_matches_jax():
    start, gates, quats, poles = track_arrays((0.0, 1.5, 3.0, 0.0), [[0.5, -0.5, 1.5]])
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    moved = gates + np.array([0.1, -0.1, 0.05], dtype=np.float32)
    args = (line, index, SEQUENCE, gates, quats, moved, quats, poles)
    from_np = repair_line(np, *args)
    from_jp = repair_line(jp, *(jp.asarray(x) for x in args))
    np.testing.assert_allclose(from_np, np.asarray(from_jp), atol=0.01)


@pytest.mark.unit
def test_pole_hazard_places_a_pole_on_a_gate_axis():
    _, gates, quats, poles = track_arrays((0.0, 1.5, 3.0, 0.5))
    hazard = np.asarray(
        add_pole_hazard(
            jax.random.PRNGKey(3), jp.asarray(gates), jp.asarray(quats), jp.asarray(poles)
        )
    )
    moved = np.flatnonzero(np.any(hazard != poles, axis=1))
    assert len(moved) == 1
    assert hazard[moved[0], 2] == poles[moved[0], 2]
    local = R.from_quat(quats).inv().apply(hazard[moved[0]] - gates)
    on_axis = (np.abs(np.abs(local[:, 0]) - 0.625) <= 0.176) & (np.abs(local[:, 1]) <= 0.121)
    assert on_axis.any()


@pytest.mark.unit
@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_line_bends_past_a_pole_right_behind_a_gate(side: float):
    """A pole on the exit axis is passed on the side of the next gate, without doubling back."""
    gates = np.array(
        [[0.0, 0.0, 0.7], [-0.5, side * 1.2, 1.2], [1.5, side * 1.2, 0.7], [1.5, 0.0, 1.2]],
        dtype=np.float32,
    )
    yaws = np.array([[0.0], [0.0], [0.0], [np.pi]])
    quats = R.from_euler("z", yaws).as_quat().astype(np.float32)
    pole = [0.5, 0.0, 1.5]
    poles = np.array([pole, [3.0, 3.0, 1.5], [3.0, -3.0, 1.5], [-3.0, 3.0, 1.5]], dtype=np.float32)
    start = np.array([-1.5, 0.0, 0.0], dtype=np.float32)
    line, index = plan_line(np, start, gates, quats, poles, SEQUENCE)
    piece = line[index[0] : index[1]]
    assert np.linalg.norm(piece[:, :2] - np.array(pole[:2]), axis=-1).min() > 0.2
    # The line leaves on the side of the next gate and never crosses to the other side
    assert np.max(-side * piece[:, 1]) < 0.1
