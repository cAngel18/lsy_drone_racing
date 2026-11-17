"""Model predictive path integral (MPPI) controller for random tracks (level 3).

A sampling-based MPC as presented in Lecture 3 (L3 Algorithms, "Sampling-Based Model Predictive
Control"). Every control step it

1. samples ``N_SAMPLES`` control sequences around the current nominal sequence,
2. rolls each of them out over ``HORIZON`` steps with a model of the drone,
3. scores every rollout with a cost that does not have to be convex or smooth: tracking of a planned
   path at a planned speed, collisions with gates, poles, the floor and the arena limits, and the
   effort and smoothness of the inputs,
4. averages the sequences with the exponential weights ``w_i = exp(-(S_i - min S) / lambda)``,
5. applies the first input and shifts the sequence by one step as the next nominal sequence.

The algorithm follows the references of the lecture:

* G. Williams, A. Aldrich, E. A. Theodorou, "Model Predictive Path Integral Control: From Theory to
  Parallel Computation", Journal of Guidance, Control, and Dynamics, 2017. Exponentially weighted
  averaging of all samples with the temperature ``TEMPERATURE`` and a constant covariance.
* C. Pinneri et al., "Sample-efficient Cross-Entropy Method for Real-time Planning", CoRL 2021.
  The previous solution is shifted as a warm start, and the noise is temporally correlated.
* P. Kicki, "Low-pass sampling in Model Predictive Path Integral Control", ICRA 2026. The sampled
  perturbations are low-pass filtered along the horizon (``NOISE_SMOOTHING``), so the inputs stay
  smooth without an extra smoothing step.

The rollout model is the identified second-order attitude model of crazyflow (``so_rpy``, also in
learnsyslab/drone-models): roll and pitch follow their commands as damped second-order systems, and
the collective thrust acts along the body z-axis. Yaw is held at zero. The thrust gain is estimated
online, because the mass of the drone is randomised and the gain of the identified model is about
5 % lower than that of the simulator's ``first_principles`` dynamics. All rollouts run in one jitted
JAX function.

The path that MPPI tracks is planned geometrically, see `plan_path`: a spline through each gate
along its normal and around the poles and gate frames that are in the way, pushed out of their
clearance and into the arena. It is replanned whenever a gate or pole enters the sensor range and
reveals its true pose, and keeps the side on which it passes each object. MPPI itself scores
collisions against the poses as they are currently observed, so the path only has to pass the
objects on a sensible side, and MPPI keeps the clearance.

Status, through the evaluation script on new random level 3 tracks (120 runs): 93 % finished,
mean lap 8.0 s, about 10 ms per control step on a laptop CPU. More reliable but slower: with
``SPEED_MAX = 3.5``, ``ACC_LATERAL = 10`` and ``ACC_FORWARD = ACC_BRAKE = 8``, 96 % finish with a
mean lap of 8.85 s (160 runs). The tracking lags the planned speed by up to a third, because the
speed profile assumes that the acceleration can change instantly. A reference that runs ahead of
the drone closes that gap but cuts corners: 0.1 s ahead gave 8.2 s at 83 %. It has only been flown
in simulation, and roll and pitch are limited per axis, so diagonal tilts reach 1.4 rad.

Requires ``control_mode = "attitude"``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

G = 9.81  # [m/s^2]
# Sampling
N_SAMPLES = 1024  # Control sequences sampled per step
HORIZON = 30  # Steps of the rollouts, at the environment frequency
TEMPERATURE = 2.0  # lambda of the exponential weights, in units of the cost
NOISE_STD = (0.15, 0.15, 0.06)  # Perturbations of roll, pitch [rad] and thrust [N]
NOISE_SMOOTHING = 0.8  # Correlation of the perturbations between steps (0: white, 1: constant)
MAX_TILT = 1.0  # [rad] limit of the roll and pitch commands
# Speed along the planned path
SPEED_MAX = 4.0  # [m/s] speed on straight parts of the path
ACC_LATERAL = 12.0  # [m/s^2] lateral acceleration that sets the speed in curves
ACC_FORWARD = 10.0  # [m/s^2] acceleration along the path
ACC_BRAKE = 10.0  # [m/s^2] deceleration along the path
REFERENCE_LEAD = 0.0  # [s] the reference starts this far ahead of the drone on the planned schedule
CURVATURE_CHORD = 4  # Samples to each side over which the curvature is measured
CURVATURE_WINDOW = 4  # Samples before and after a curve that it slows down
# Cost weights, per step of the horizon
W_LATERAL = 60.0  # Distance from the reference across the path [1/m^2]
W_LAG = 15.0  # Distance from the reference along the path [1/m^2]
W_VEL = 1.0  # Velocity error [s^2/m^2]
W_TERMINAL = 5.0  # Factor of the tracking costs of the last step
W_TILT = 0.5  # Roll and pitch commands [1/rad^2]
W_THRUST = 20.0  # Thrust command relative to hover [1/N^2]
W_RATE = 20.0  # Change of the commands between steps [1/rad^2, 1/N^2]
W_OBSTACLE = 3000.0  # Penetration of the safety margin of gates and poles [1/m^2]
W_CRASH = 300.0  # Every step after a predicted collision
W_FLOOR = 3000.0  # Height below FLOOR_HEIGHT [1/m^2]
W_ARENA = 3000.0  # Distance outside of the arena limits [1/m^2]
# Geometry of the drone and the track, from the collision models
DRONE_RADIUS = 0.086  # [m] collision sphere of the drone
POLE_RADIUS = 0.015  # [m]
SAFETY_MARGIN = 0.06  # [m] distance from gates and poles at which the cost starts to rise
FLOOR_HEIGHT = 0.08  # [m]
ARENA_MARGIN = 0.1  # [m] margin inside the safety limits of the track
# Collision boxes of a gate in its own frame: top, bottom, left and right bar, and the stand
GATE_BOX_POS = np.array(
    [[0.0, 0.0, 0.28], [0.0, 0.0, -0.28], [0.0, -0.28, 0.0], [0.0, 0.28, 0.0], [0.0, 0.0, -0.86]]
)
GATE_BOX_HALF = np.array(
    [
        [0.01, 0.36, 0.08],
        [0.01, 0.36, 0.08],
        [0.01, 0.08, 0.36],
        [0.01, 0.08, 0.36],
        [0.05, 0.05, 0.5],
    ]
)
# Path planning
PATH_DS = 0.05  # [m] distance between the samples of the path
TAKEOFF_HEIGHT = 0.3  # [m] rise before the path turns towards the first gate
GATE_APPROACH = 0.35  # [m] the path enters a gate along its normal from this distance
GATE_EXIT = 0.35  # [m] and leaves it along its normal for this distance
GATE_WALL = 0.55  # [m] half width of a gate as a wall for detours: frame, drone and a margin
GATE_TOP = 0.55  # [m] height above the centre of a gate at which the path may cross it
POLE_CLEARANCE = 0.3  # [m] distance of the path from poles, MPPI handles the pole randomisation
GATE_CLEARANCE = 0.25  # [m] distance of the path from gate frames, outside of the gate passes
ARENA_CLEARANCE = 0.25  # [m] distance of the path from the arena limits
DETOUR_MARGIN = 0.1  # [m] detour points lie this far beyond the clearance of the object
DETOUR_DEPTH = 2  # Nested detours per piece between two gates
DETOUR_SLIDE = 20  # Steps of PATH_DS by which a detour point may slide out beyond other poles
PATH_PIN = 0.25  # [m] around a gate the path stays as the spline runs, straight through the gate
PATH_TAPER = 0.3  # [m] beyond that, pushing and smoothing set in gradually over this distance
PATH_SMOOTHING = 3  # Smoothing iterations after each push out of the objects
PATH_PUSH = 10  # Iterations that push the smoothed path out of objects and into the arena
PATH_EXTENSION = 2.0  # [m] the path continues straight on after the last gate
# Online estimate of the thrust gain (recursive least squares)
GAIN_PRIOR_WEIGHT = 25.0  # Weight of the identified gain, in samples
GAIN_FORGET = 0.99  # Forgetting factor per step
GAIN_BOUNDS = (0.8, 1.25)  # Bounds relative to the identified gain
GAIN_THRUST_LAG = 0.5  # Per step, first-order lag from the commanded to the realised thrust
GAIN_MIN_HEIGHT = 0.08  # [m] no updates close to the ground


def gate_axes(gates_quat: NDArray) -> tuple[NDArray, NDArray]:
    """Horizontal normals (g, 3) and lateral axes (g, 2) of gates."""
    normal = R.from_quat(gates_quat).as_matrix()[:, :, 0]
    normal[:, 2] = 0.0
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9
    return normal, np.stack([-normal[:, 1], normal[:, 0]], axis=-1)


def first_blocker(
    a: NDArray,
    b: NDArray,
    gates_pos: NDArray,
    lateral: NDArray,
    poles: NDArray,
    arena: tuple[NDArray, NDArray],
    sides: dict[tuple, tuple[NDArray, frozenset]],
    piece: int,
) -> NDArray | None:
    """The detour point around the first object in the way of the straight piece from a to b.

    Poles are vertical lines, gates are vertical walls of half width GATE_WALL up to GATE_TOP above
    their centre. An object is passed on the side with the shorter path the first time it is in
    the way. That side is stored in ``sides``, so that a replan does not switch to the other side
    when the object turns out to be somewhere else than its nominal position. A detour point that
    would lie too close to another pole slides further out, beyond that pole.

    Args:
        a: The start of the piece (3,).
        b: The end of the piece (3,).
        gates_pos: The gate positions (g, 3).
        lateral: The horizontal lateral axes of the gates (g, 2).
        poles: The pole positions (o, 3).
        arena: The lower and upper limits of the arena (3,) each.
        sides: The direction in which each piece detours around each object so far, and the poles
            beyond which the detour point slid.
        piece: The index of the piece in the gate order.

    Returns:
        The detour point (3,), or None if the piece is clear.
    """
    d = b[:2] - a[:2]
    length2 = float(d @ d) + 1e-9

    def place(center: NDArray, offset: NDArray, beyond: frozenset) -> tuple[NDArray, frozenset]:
        """Slide a detour point further out until it is clear of all poles.

        It also slides past the poles in ``beyond``, if they are still in its way, which keeps the
        decision of an earlier plan to go around them. Returns the point and the poles it passed.
        """
        direction = offset / np.linalg.norm(offset)
        point, passed = center + offset, set()
        for _ in range(DETOUR_SLIDE):
            rel = point - poles[:, :2]
            near = np.linalg.norm(rel, axis=-1) < POLE_CLEARANCE
            across = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
            ahead = (rel @ direction < 0.0) & (across < POLE_CLEARANCE + DETOUR_MARGIN)
            ahead &= np.isin(np.arange(len(poles)), list(beyond))
            if not np.any(near | ahead):
                break
            passed |= set(np.flatnonzero(near).tolist())
            point = point + direction * PATH_DS
        return point, frozenset(passed) | beyond

    def detour(key: tuple, center: NDArray, offset: NDArray) -> NDArray:
        if key not in sides:
            # The shorter way around, unless it leaves the arena
            low, high = arena[0][:2] + ARENA_CLEARANCE, arena[1][:2] - ARENA_CLEARANCE
            best = None
            for s in (1.0, -1.0):
                p, passed = place(center, s * offset, frozenset())
                cost = np.linalg.norm(p - a[:2]) + np.linalg.norm(p - b[:2])
                cost += 10.0 * np.sum(np.maximum(low - p, 0.0) + np.maximum(p - high, 0.0))
                if best is None or cost < best[0]:
                    best = (cost, s * offset / np.linalg.norm(offset), passed)
            sides[key] = best[1:]
        # The side is kept as a direction in the world frame, which does not change with the piece
        direction, beyond = sides[key]
        point, passed = place(center, np.sign(offset @ direction + 1e-12) * offset, beyond)
        sides[key] = (direction, passed)
        return point

    hits = []
    side = np.array([-d[1], d[0]]) / np.sqrt(length2)
    for j, pole in enumerate(poles):
        t = np.clip((pole[:2] - a[:2]) @ d / length2, 0.0, 1.0)
        if np.linalg.norm(a[:2] + t * d - pole[:2]) < POLE_CLEARANCE:
            key = (piece, "pole", j)
            hits.append((t, detour(key, pole[:2], side * (POLE_CLEARANCE + DETOUR_MARGIN))))
    for j, (gate, lat) in enumerate(zip(gates_pos, lateral)):
        wall = lat * GATE_WALL
        denominator = d[0] * wall[1] - d[1] * wall[0]
        if abs(denominator) < 1e-9:
            continue
        rel = gate[:2] - a[:2]
        t = (rel[0] * wall[1] - rel[1] * wall[0]) / denominator
        u = (rel[0] * d[1] - rel[1] * d[0]) / denominator
        height = a[2] + t * (b[2] - a[2])
        if 0.0 <= t <= 1.0 and abs(u) <= 1.0 and height < gate[2] + GATE_TOP:
            key = (piece, "gate", j)
            hits.append((t, detour(key, gate[:2], lat * (GATE_WALL + DETOUR_MARGIN))))
    if not hits:
        return None
    t, xy = min(hits, key=lambda hit: hit[0])
    return np.array([xy[0], xy[1], a[2] + t * (b[2] - a[2])])


def detours(
    a: NDArray,
    b: NDArray,
    gates_pos: NDArray,
    lateral: NDArray,
    poles: NDArray,
    arena: tuple[NDArray, NDArray],
    sides: dict[tuple, tuple[NDArray, frozenset]],
    piece: int,
    depth: int = DETOUR_DEPTH,
) -> list[NDArray]:
    """Detour points that lead the straight piece from a to b around the objects in its way."""
    point = first_blocker(a, b, gates_pos, lateral, poles, arena, sides, piece)
    if point is None or depth == 0:
        return []
    args = (gates_pos, lateral, poles, arena, sides, piece, depth - 1)
    return detours(a, point, *args) + [point] + detours(point, b, *args)


def push_out(
    path: NDArray,
    gates_pos: NDArray,
    gates_quat: NDArray,
    poles: NDArray,
    arena: tuple[NDArray, NDArray],
) -> NDArray:
    """Move path samples that are too close to poles or gate frames away from them, into the arena.

    Each sample moves directly away from the object, so the path keeps the side on which it passes.
    """
    rel = path[:, None, :2] - poles[None, :, :2]
    dist = np.linalg.norm(rel, axis=-1, keepdims=True) + 1e-9
    push = np.sum(rel / dist * np.maximum(POLE_CLEARANCE - dist, 0.0), axis=1)
    path = path + np.concatenate([push, np.zeros_like(push[:, :1])], axis=-1)
    # Closest points of the gate boxes, in the frames of the gates (l, g, b, 3)
    rot = R.from_quat(gates_quat).as_matrix()
    local = np.einsum("gji,lgj->lgi", rot, path[:, None] - gates_pos[None])[:, :, None]
    offset = np.clip(local, GATE_BOX_POS - GATE_BOX_HALF, GATE_BOX_POS + GATE_BOX_HALF) - local
    dist2 = np.sum(offset**2, axis=-1).reshape(len(path), -1)
    closest = np.argmin(dist2, axis=1)
    gate = closest // len(GATE_BOX_POS)
    away = -offset.reshape(len(path), -1, 3)[np.arange(len(path)), closest]  # Gate frame
    away = np.einsum("lij,lj->li", rot[gate], away)  # World frame
    dist = np.linalg.norm(away, axis=-1, keepdims=True) + 1e-9
    path = path + away / dist * np.maximum(GATE_CLEARANCE - dist, 0.0)
    return np.clip(path, arena[0] + ARENA_CLEARANCE, arena[1] - ARENA_CLEARANCE)


def plan_path(
    start: NDArray,
    gates_pos: NDArray,
    gates_quat: NDArray,
    poles: NDArray,
    sequence: NDArray,
    arena: tuple[NDArray, NDArray],
    sides: dict[tuple, tuple[NDArray, frozenset]],
) -> tuple[NDArray, NDArray]:
    """Plan a smooth path from the start through the gates in their order.

    The path rises from the start, enters every gate along its normal from GATE_APPROACH in front
    of it and leaves it along its normal for GATE_EXIT. The pieces in between detour around poles
    and gate frames, which includes a loop around a gate that has to be entered from behind. A cubic
    Hermite spline joins these waypoints, with the gate normal as the direction at the gates. It
    passes through every waypoint, so a sharp turn becomes a loop and does not collapse into a cusp.
    Finally the path is pushed out of the clearance of the objects and into the arena.

    Args:
        start: The start position (3,).
        gates_pos: The gate positions (g, 3).
        gates_quat: The gate orientations as xyzw quaternions (g, 4).
        poles: The pole positions (o, 3).
        sequence: The gate IDs of the gate order (m,).
        arena: The lower and upper limits of the arena (3,) each.
        sides: The sides on which objects are passed, see `first_blocker`. Updated in place.

    Returns:
        The path samples (l, 3), spaced PATH_DS apart, and the sample index of each gate pass (m,).
    """
    normal, lateral = gate_axes(gates_quat)
    points = [start, start + np.array([0.0, 0.0, TAKEOFF_HEIGHT])]
    directions = {0: np.array([0.0, 0.0, 1.0])}  # Fixed directions of the path at waypoints
    gate_points = []
    low, high = arena[0] + ARENA_CLEARANCE, arena[1] - ARENA_CLEARANCE

    def reach(point: NDArray, direction: NDArray, distance: float) -> float:
        """How far a point can move along a horizontal direction and stay in the arena."""
        with np.errstate(divide="ignore", invalid="ignore"):
            limit = np.where(direction > 0, (high - point) / direction, (low - point) / direction)
        limit = np.where(np.abs(direction) > 1e-9, limit, np.inf)[:2]
        return float(np.clip(limit.min(), 0.1, distance))

    for piece, gate in enumerate(sequence):
        center, n = gates_pos[gate], normal[gate]
        enter = center - reach(center, -n, GATE_APPROACH) * n
        points += detours(points[-1], enter, gates_pos, lateral, poles, arena, sides, piece)
        gate_points.append(len(points) + 1)
        directions[len(points) + 1] = n
        points += [enter, center, center + reach(center, n, GATE_EXIT) * n]
    points.append(points[-1] + PATH_EXTENSION * normal[sequence[-1]])
    points = np.asarray(points)

    # Spline over the chord length. At the gates the path points along the gate normal, elsewhere
    # from the waypoint before to the one after (Catmull-Rom). The entry and exit points do not fix
    # the direction, so that a turn right after a gate is a loop and not a hook.
    chord = np.maximum(np.linalg.norm(np.diff(points, axis=0), axis=1), 1e-3)
    knots = np.concatenate([[0.0], np.cumsum(chord)])
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[-1] = points[-1] - points[-2]
    for i, direction in directions.items():
        tangents[i] = direction
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    dense = CubicHermiteSpline(knots, points, tangents)(
        np.linspace(0.0, knots[-1], 20 * len(points) + int(knots[-1] / 0.01))
    )
    marks = np.searchsorted(np.linspace(0.0, knots[-1], len(dense)), knots[gate_points])

    def resample(line: NDArray, marks: NDArray) -> tuple[NDArray, NDArray]:
        """Sample a polyline at a constant distance, and convert sample marks along with it."""
        length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
        grid = np.arange(0.0, length[-1], PATH_DS)
        samples = np.stack([np.interp(grid, length, line[:, i]) for i in range(3)], axis=-1)
        return samples, np.round(length[marks] / PATH_DS).astype(int)

    path, gate_index = resample(dense, marks)
    # The parts around the gates stay in place, and the strength of the corrections rises gradually
    # away from them, so that they do not leave a kink there
    index = np.arange(len(path))
    from_gate = np.min(np.abs(index[:, None] - gate_index[None]), axis=1) * PATH_DS
    strength = np.clip((from_gate - PATH_PIN) / PATH_TAPER, 0.0, 1.0)
    strength[[0, -1]] = 0.0
    strength = strength[:, None]

    def smooth(path: NDArray) -> NDArray:
        target = path.copy()
        target[1:-1] = 0.5 * (path[:-2] + path[2:])
        return path + 0.5 * strength * (target - path)

    for _ in range(PATH_PUSH):
        pushed = push_out(path, gates_pos, gates_quat, poles, arena)
        path = path + (strength > 0.0) * (pushed - path)
        for _ in range(PATH_SMOOTHING):
            path = smooth(path)
    # The corrections change the distances between the samples, so they are made regular again
    return resample(path, gate_index)


def speed_profile(path: NDArray) -> tuple[NDArray, NDArray]:
    """Speed at each sample of a path and the time at which it is reached.

    The speed is limited by SPEED_MAX, by the lateral acceleration in curves, and by the forward
    and braking accelerations along the path, starting from rest.

    Args:
        path: The path samples (l, 3).

    Returns:
        The speeds (l,) and the arrival times (l,).
    """
    seg = np.linalg.norm(np.diff(path, axis=0), axis=-1) + 1e-6
    # Curvature from the turn of the chords to the samples CURVATURE_CHORD before and after, so
    # that small kinks between neighbouring samples do not count as tight curves
    k = CURVATURE_CHORD
    padded = np.concatenate([np.repeat(path[:1], k, 0), path, np.repeat(path[-1:], k, 0)])
    back, ahead = path - padded[: -2 * k], padded[2 * k :] - path
    size_back = np.linalg.norm(back, axis=-1) + 1e-9
    size_ahead = np.linalg.norm(ahead, axis=-1) + 1e-9
    turn = np.linalg.norm(ahead / size_ahead[:, None] - back / size_back[:, None], axis=-1)
    curvature = turn / (0.5 * (size_back + size_ahead))
    padded = np.pad(curvature, CURVATURE_WINDOW, mode="edge")
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * CURVATURE_WINDOW + 1)
    speed = np.minimum(SPEED_MAX, np.sqrt(ACC_LATERAL / (window.max(axis=-1) + 1e-6)))
    speed[0] = 0.0
    for i in range(1, len(path)):
        speed[i] = min(speed[i], np.sqrt(speed[i - 1] ** 2 + 2 * ACC_FORWARD * seg[i - 1]))
    for i in range(len(path) - 2, -1, -1):
        speed[i] = min(speed[i], np.sqrt(speed[i + 1] ** 2 + 2 * ACC_BRAKE * seg[i]))
    mean_speed = np.maximum(0.5 * (speed[1:] + speed[:-1]), 0.2)
    return speed, np.concatenate([[0.0], np.cumsum(seg / mean_speed)])


def body_z(rpy: Any) -> Any:
    """Body z-axis in the world frame for extrinsic xyz Euler angles (..., 3)."""
    cr, sr = jnp.cos(rpy[..., 0]), jnp.sin(rpy[..., 0])
    cp, sp = jnp.cos(rpy[..., 1]), jnp.sin(rpy[..., 1])
    cy, sy = jnp.cos(rpy[..., 2]), jnp.sin(rpy[..., 2])
    return jnp.stack([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr], axis=-1)


def model_step(state: Any, u: Any, gain: Any, coef: dict[str, Any], dt: float) -> Any:
    """One step of the identified attitude model for a batch of states.

    Args:
        state: The states [pos, rpy, vel, rpy_rates] (..., 12).
        u: The commands [roll, pitch, thrust] (..., 3). The yaw command is zero.
        gain: The acceleration per Newton of commanded thrust [1/kg].
        coef: The second-order attitude coefficients ``rpy_coef``, ``rpy_rates_coef`` and
            ``cmd_rpy_coef`` (3,) each.
        dt: The step duration [s].

    Returns:
        The next states (..., 12).
    """
    pos, rpy, vel, rates = state[..., :3], state[..., 3:6], state[..., 6:9], state[..., 9:]
    cmd_rpy = jnp.concatenate([u[..., :2], jnp.zeros_like(u[..., :1])], axis=-1)
    acc = gain * u[..., 2:] * body_z(rpy) - jnp.asarray([0.0, 0.0, G])
    rates_dot = coef["rpy_coef"] * rpy + coef["rpy_rates_coef"] * rates
    rates = rates + dt * (rates_dot + coef["cmd_rpy_coef"] * cmd_rpy)
    rpy = rpy + dt * rates
    return jnp.concatenate([pos + dt * vel, rpy, vel + dt * acc, rates], axis=-1)


def sample_noise(key: Any, n: int, horizon: int) -> Any:
    """Low-pass filtered Gaussian perturbations (n, horizon, 3) with the variance of NOISE_STD."""
    white = jax.random.normal(key, (horizon, n, 3)) * jnp.asarray(NOISE_STD)
    beta, scale = NOISE_SMOOTHING, np.sqrt(1.0 - NOISE_SMOOTHING**2)

    def low_pass(previous: Any, w: Any) -> tuple[Any, Any]:
        current = beta * previous + scale * w
        return current, current

    _, noise = jax.lax.scan(low_pass, white[0], white[1:])
    return jnp.swapaxes(jnp.concatenate([white[:1], noise]), 0, 1)


def obstacle_gap(pos: Any, scene: dict[str, Any]) -> Any:
    """Distance from the drone surface to the closest gate box or pole (n,)."""
    # Positions in the frame of each gate (n, g, 3), the rotation columns are the gate axes
    local = jnp.einsum("gji,ngj->ngi", scene["gates_rot"], pos[:, None] - scene["gates_pos"][None])
    local = local[:, :, None]  # (n, g, 1, 3)
    low = jnp.asarray(GATE_BOX_POS - GATE_BOX_HALF)
    high = jnp.asarray(GATE_BOX_POS + GATE_BOX_HALF)
    offset = jnp.clip(local, low, high) - local  # (n, g, b, 3)
    gate_dist = jnp.sqrt(jnp.min(jnp.sum(offset**2, axis=-1), axis=(1, 2)) + 1e-9)
    pole_rel = pos[:, None, :2] - scene["poles"][None, :, :2]
    pole_dist = jnp.sqrt(jnp.min(jnp.sum(pole_rel**2, axis=-1), axis=-1) + 1e-9) - POLE_RADIUS
    return jnp.minimum(gate_dist, pole_dist) - DRONE_RADIUS


def stage_cost(
    state: Any, u: Any, u_before: Any, ref: tuple[Any, Any, Any], scene: dict[str, Any], hover: Any
) -> tuple[Any, Any, Any]:
    """Cost of one step of the rollouts, without the crash cost that depends on earlier steps.

    Args:
        state: The predicted states after the step (n, 12).
        u: The commands of the step (n, 3).
        u_before: The commands of the step before (n, 3).
        ref: The reference position, unit tangent and velocity of the step (3,) each.
        scene: The observed ``gates_pos``, ``gates_rot``, ``poles`` and the arena limits ``low``
            and ``high``.
        hover: The hover thrust [N].

    Returns:
        The tracking cost, the remaining cost and the collision flags (n,) each.
    """
    pos, vel = state[..., :3], state[..., 6:9]
    ref_pos, ref_tangent, ref_vel = ref
    error = pos - ref_pos
    lag = jnp.sum(error * ref_tangent, axis=-1)
    lateral2 = jnp.maximum(jnp.sum(error**2, axis=-1) - lag**2, 0.0)
    tracking = W_LATERAL * lateral2 + W_LAG * lag**2
    tracking = tracking + W_VEL * jnp.sum((vel - ref_vel) ** 2, axis=-1)

    gap = obstacle_gap(pos, scene)
    safety = W_OBSTACLE * jnp.maximum(SAFETY_MARGIN - gap, 0.0) ** 2
    safety = safety + W_FLOOR * jnp.maximum(FLOOR_HEIGHT - pos[:, 2], 0.0) ** 2
    outside = jnp.maximum(scene["low"] - pos, 0.0) + jnp.maximum(pos - scene["high"], 0.0)
    safety = safety + W_ARENA * jnp.sum(outside**2, axis=-1)

    effort = W_TILT * jnp.sum(u[:, :2] ** 2, axis=-1) + W_THRUST * (u[:, 2] - hover) ** 2
    effort = effort + W_RATE * jnp.sum((u - u_before) ** 2, axis=-1)
    collided = (gap < 0.0) | (pos[:, 2] < 0.0)
    return tracking, safety + effort, collided


@partial(jax.jit, static_argnames=("dt",))
def mppi_step(
    key: Any,
    state0: Any,
    u_nominal: Any,
    u_applied: Any,
    refs: tuple[Any, Any, Any],
    scene: dict[str, Any],
    gain: Any,
    bounds: tuple[Any, Any],
    coef: dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    """One MPPI update of the nominal control sequence.

    Args:
        key: The random key of this step.
        state0: The current state [pos, rpy, vel, rpy_rates] (12,).
        u_nominal: The nominal control sequence [roll, pitch, thrust] (HORIZON, 3).
        u_applied: The command applied in the last step (3,).
        refs: The reference positions, unit tangents and velocities (HORIZON, 3) each.
        scene: The observed objects and the arena limits, see `stage_cost`.
        gain: The estimated acceleration per Newton of thrust [1/kg].
        bounds: The lower and upper bounds of the commands (3,) each.
        coef: The attitude model coefficients, see `model_step`.
        dt: The step duration [s].

    Returns:
        The updated sequence ``u``, the positions predicted for it ``path``, the lowest cost
        ``cost_min`` and the effective number of samples ``n_eff``.
    """
    noise = sample_noise(key, N_SAMPLES, HORIZON)
    noise = noise.at[0].set(0.0)  # The nominal sequence itself stays a candidate
    controls = jnp.clip(u_nominal[None] + noise, bounds[0], bounds[1])  # (n, h, 3)
    first = jnp.broadcast_to(u_applied, (N_SAMPLES, 1, 3))
    previous = jnp.concatenate([first, controls[:, :-1]], axis=1)
    weight = jnp.ones(HORIZON).at[-1].set(W_TERMINAL)
    hover = G / gain

    def step(carry: tuple[Any, Any, Any], xs: tuple[Any, ...]) -> tuple[Any, None]:
        states, total, crashed = carry
        u, u_before, ref_pos, ref_tangent, ref_vel, w = xs
        states = model_step(states, u, gain, coef, dt)
        tracking, other, collided = stage_cost(
            states, u, u_before, (ref_pos, ref_tangent, ref_vel), scene, hover
        )
        crashed = crashed | collided
        return (states, total + w * tracking + other + W_CRASH * crashed, crashed), None

    init = (
        jnp.broadcast_to(state0, (N_SAMPLES, 12)),
        jnp.zeros(N_SAMPLES),
        jnp.zeros(N_SAMPLES, dtype=bool),
    )
    xs = (jnp.swapaxes(controls, 0, 1), jnp.swapaxes(previous, 0, 1), *refs, weight)
    (_, costs, _), _ = jax.lax.scan(step, init, xs)

    # Exponential weights of the path integral, shifted by the minimum for numerical stability
    weights = jnp.exp(-(costs - jnp.min(costs)) / TEMPERATURE)
    weights = weights / jnp.sum(weights)
    u_new = jnp.sum(weights[:, None, None] * controls, axis=0)

    def predict(state: Any, u: Any) -> tuple[Any, Any]:
        state = model_step(state, u, gain, coef, dt)
        return state, state[:3]

    _, path = jax.lax.scan(predict, state0, u_new)
    return {
        "u": u_new,
        "path": path,
        "cost_min": jnp.min(costs),
        "n_eff": 1.0 / jnp.sum(weights**2),
    }


class ThrustGainEstimator:
    """Recursive least squares estimate of the acceleration per Newton of commanded thrust.

    The model is ``(dv/dt + g e_z) . z_body = gain * thrust``, where the thrust follows the commands
    with a first-order lag. The identified gain is the prior.
    """

    def __init__(self, gain: float, thrust: float, thrust_range: tuple[float, float]):
        """Start from the identified gain and the initial thrust."""
        self.prior = gain
        self._num = GAIN_PRIOR_WEIGHT * gain * thrust**2
        self._den = GAIN_PRIOR_WEIGHT * thrust**2
        self._thrust = thrust
        low, high = thrust_range
        self._active = (low + 0.15 * (high - low), low + 0.85 * (high - low))

    @property
    def gain(self) -> float:
        """The current estimate, within GAIN_BOUNDS of the prior."""
        ratio = np.clip(self._num / self._den / self.prior, *GAIN_BOUNDS)
        return float(ratio * self.prior)

    def update(self, pos: NDArray, quat: NDArray, acc: NDArray, thrust_cmd: float):
        """Add the acceleration measured over the last step, in which thrust_cmd was commanded."""
        self._thrust += GAIN_THRUST_LAG * (thrust_cmd - self._thrust)
        if pos[2] < GAIN_MIN_HEIGHT or not self._active[0] < self._thrust < self._active[1]:
            return
        z_body = R.from_quat(quat).as_matrix()[:, 2]
        y = float((acc + np.array([0.0, 0.0, G])) @ z_body)
        self._num = GAIN_FORGET * self._num + self._thrust * y
        self._den = GAIN_FORGET * self._den + self._thrust**2


class MPPIRacer(Controller):
    """MPPI controller that tracks an online planned path through random tracks."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load the model parameters and plan the path through the nominal track.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._dt = 1.0 / config.env.freq
        params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._coef = {
            k: jnp.asarray(params[k], dtype=jnp.float32)
            for k in ("rpy_coef", "rpy_rates_coef", "cmd_rpy_coef")
        }
        self._gain_identified = float(params["cmd_f_coef"]) / float(params["mass"])
        self._thrust_range = (4 * float(params["thrust_min"]), 4 * float(params["thrust_max"]))
        self._bounds = (
            jnp.asarray([-MAX_TILT, -MAX_TILT, self._thrust_range[0]], dtype=jnp.float32),
            jnp.asarray([MAX_TILT, MAX_TILT, self._thrust_range[1]], dtype=jnp.float32),
        )
        limits = config.env.track.get("safety_limits", None)
        low = np.asarray(limits["pos_limit_low"] if limits else [-3.0, -3.0, 0.0], dtype=np.float32)
        high = np.asarray(limits["pos_limit_high"] if limits else [3.0, 3.0, 2.5], dtype=np.float32)
        self._limits = (low.astype(np.float64), high.astype(np.float64))
        self._arena = (jnp.asarray(low + ARENA_MARGIN), jnp.asarray(high - ARENA_MARGIN))
        self._seed = max(int(config.env.seed), 0)
        self._start = np.asarray(obs["pos"], dtype=np.float64)
        self._gate_sequence = np.asarray(obs["gate_sequence"], dtype=int)
        self._n_passes = self._gate_sequence.shape[0]
        self.episode_reset()

    def episode_reset(self):
        """Reset the nominal sequence, the path and the thrust gain estimate."""
        self._key = jax.random.key(self._seed)
        hover = G / self._gain_identified
        self._gain = ThrustGainEstimator(self._gain_identified, hover, self._thrust_range)
        self._vel_prev = None
        self._u = np.tile(np.asarray([0.0, 0.0, hover], dtype=np.float32), (HORIZON, 1))
        self._u_applied = self._u[0].copy()
        self._path_idx = 0
        self._revealed = None
        self._sides = {}
        self.path = np.zeros((0, 3))
        self.prediction = np.zeros((HORIZON, 3))
        self.stats = {"cost_min": 0.0, "n_eff": 0.0}

    def _replan(self, obs: dict[str, NDArray[np.floating]]):
        """Plan the path through the observed track and locate the drone on it again."""
        self.path, self._gate_index = plan_path(
            self._start,
            np.asarray(obs["gates_pos"], dtype=np.float64),
            np.asarray(obs["gates_quat"], dtype=np.float64),
            np.asarray(obs["obstacles_pos"], dtype=np.float64),
            self._gate_sequence,
            self._limits,
            self._sides,
        )
        self._speed, self._time = speed_profile(self.path)
        self._path_idx = self._locate(np.asarray(obs["pos"]), int(obs["n_gates_passed"]), None)

    def _locate(self, pos: NDArray, n_passed: int, guess: int | None) -> int:
        """Index of the path sample closest to the drone, between the last and the next gate.

        Without a guess the whole piece between the gates is searched, otherwise a window around
        the guess, so that the drone cannot jump to a part of the path that crosses itself.
        """
        first = self._gate_index[n_passed - 1] if n_passed > 0 else 0
        last = self._gate_index[n_passed] if n_passed < self._n_passes else len(self.path) - 1
        if guess is not None:
            first, last = max(first, guess - 5), max(min(last, guess + 10), first)
        candidates = np.arange(first, last + 1)
        distance = np.linalg.norm(self.path[candidates] - pos, axis=-1)
        return int(candidates[np.argmin(distance)])

    def _reference(self, pos: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Reference positions, tangents and velocities of the next HORIZON steps.

        The reference starts at the projection of the drone on the path and moves along it as fast
        as the speed profile prescribes.
        """
        i = self._path_idx
        j = min(i + 1, len(self.path) - 1)
        tangent = self.path[j] - self.path[i]
        along = (pos - self.path[i]) @ tangent / (tangent @ tangent + 1e-9)
        progress = i + np.clip(along, 0.0, 1.0)
        index = np.arange(len(self.path), dtype=np.float64)
        t = np.interp(progress, index, self._time) + REFERENCE_LEAD
        t = t + self._dt * np.arange(1, HORIZON + 1)
        s = np.interp(t, self._time, index)
        ref_pos = np.stack([np.interp(s, index, self.path[:, k]) for k in range(3)], axis=-1)
        lower = np.clip(np.floor(s).astype(int), 0, len(self.path) - 2)
        tangents = self.path[lower + 1] - self.path[lower]
        tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True) + 1e-9
        speed = np.interp(s, index, self._speed)
        return ref_pos, tangents, speed[:, None] * tangents

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Run one MPPI update and return its first command.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information.

        Returns:
            The attitude command [roll, pitch, yaw, thrust].
        """
        pos, quat, vel = (np.asarray(obs[k], dtype=np.float64) for k in ("pos", "quat", "vel"))
        if self._vel_prev is not None:
            self._gain.update(pos, quat, (vel - self._vel_prev) / self._dt, self._u_applied[2])
        self._vel_prev = vel

        revealed = np.concatenate([obs["gates_visited"], obs["obstacles_visited"]])
        if self._revealed is None or np.any(revealed != self._revealed):
            self._revealed = revealed
            self._replan(obs)
        self._path_idx = self._locate(pos, int(obs["n_gates_passed"]), self._path_idx)

        rpy = R.from_quat(quat).as_euler("xyz")
        rates = ang_vel2rpy_rates(quat, np.asarray(obs["ang_vel"], dtype=np.float64))
        scene = {
            "gates_pos": jnp.asarray(obs["gates_pos"], dtype=jnp.float32),
            "gates_rot": jnp.asarray(R.from_quat(obs["gates_quat"]).as_matrix(), dtype=jnp.float32),
            "poles": jnp.asarray(obs["obstacles_pos"], dtype=jnp.float32),
            "low": self._arena[0],
            "high": self._arena[1],
        }
        refs = tuple(jnp.asarray(r, dtype=jnp.float32) for r in self._reference(pos))
        self._key, key = jax.random.split(self._key)
        out = mppi_step(
            key,
            jnp.asarray(np.concatenate([pos, rpy, vel, rates]), dtype=jnp.float32),
            jnp.asarray(self._u),
            jnp.asarray(self._u_applied),
            refs,
            scene,
            jnp.float32(self._gain.gain),
            self._bounds,
            self._coef,
            dt=self._dt,
        )
        u = np.asarray(out["u"])
        self.prediction = np.asarray(out["path"])
        self.stats = {"cost_min": float(out["cost_min"]), "n_eff": float(out["n_eff"])}
        self._u_applied = u[0].copy()
        self._u = np.concatenate([u[1:], u[-1:]])  # Warm start for the next step
        return np.asarray([u[0, 0], u[0, 1], 0.0, u[0, 2]], dtype=np.float32)

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

    def render_callback(self, sim: Sim):
        """Draw the planned path and the trajectory that MPPI predicts."""
        from crazyflow.sim.visualize import draw_line

        draw_line(sim, self.path[::4], rgba=(0.0, 1.0, 0.0, 1.0))
        draw_line(sim, self.prediction, rgba=(1.0, 0.0, 0.0, 1.0))
