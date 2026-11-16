"""Track-independent reinforcement-learning controller for random tracks (level 3).

Unlike [rl_racer][lsy_drone_racing.control.rl_racer], which memorises one track, this policy has to
fly tracks it has never seen. It is a planner with a learned tracker:

* **An online planner.** At the start of a race, `plan_line` plans a line through the gates from the
  observed nominal track, with detours around poles and gate frames and a loop for gates that have
  to be entered from behind. `repair_line` adapts it whenever a gate or pole is revealed, because a
  line that is clear of the nominal poses can run into the true ones. The policy observes points
  ahead on this line, so it knows the path and the curvature to come.

* **Yaw-invariant observations and actions.** Everything is expressed in the horizontal frame of the
  target gate, including the tilt command, and no absolute position is observed. A turn is the same
  problem wherever it lies on the track and however the track is oriented.
* **Objects are ordered by what they mean.** The policy sees the target gate, the gate after it, the
  remaining gates and the poles sorted by distance, not by their index in the track definition.
* **Collision geometry is computed, not learned.** The closest solid point of the gates and the
  position of nearby objects along and across the direction of travel are direct inputs. Otherwise
  a small network would have to reconstruct rotated frames and stands from their poses. Most crashes
  of the first policy without these inputs were into the gate that it had just passed.
* **A speed cap as input.** The policy is trained over a range of caps, so a single network covers
  the whole trade-off between speed and reliability, and `SPEED_CAP` selects the operating point.

Gates and obstacles are at their nominal poses until they enter the sensor range, as in the
environment. The conventions are defined here and are shared with
[train_rl_general][lsy_drone_racing.control.train_rl_general], which is why the functions take the
array namespace ``xp``. The weights are a plain ``.npz`` file, so inference only needs NumPy.

Status at `SPEED_CAP`, through ``scripts/sim.py`` on new random seeds: level 3 (500 runs) 87.6 %
finished, mean lap 8.00 s. Levels 0 and 1 (100 runs each) all finished, mean lap 6.54 s. The level 2
track (200 runs) 88.5 %, mean lap 6.61 s, although it was never trained on. On level 3, about 8 %
of the runs leave the safety box of the lab at some point, which would end a real flight. Flying
faster is possible by raising `SPEED_CAP`, but tracks unlike the training data suffer first:
mirrored level 2 tracks drop from 87-90 % at 2 m/s to about 60 % at 3.5 m/s. It has only been flown
in simulation, on tracks with four gates, four poles and the gate order of the course.

Requires ``control_mode = "attitude"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.rl_racer import (
    ANG_VEL_SCALE,
    GATE_BOX_HALF,
    GATE_BOX_POS,
    POS_SCALE,
    VEL_SCALE,
    G,
    mass_estimate_init,
    mass_estimate_update,
    mass_ratio,
    mlp_forward,
    quat_to_matrix,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

POLICY_PATH = Path(__file__).parent / "rl_general_policy.npz"
SPEED_CAP = 2.0  # [m/s] the fastest speed at which tracks the policy has not seen stay reliable
SPEED_SCALE = 5.0  # [m/s] scale of the speed cap in the observation
MASS_RATIO_SCALE = 0.12  # the mass is randomised by +-11.5 %
WALL_RANGE = 1.0  # [m] distance at which the arena limits start to appear in the observation
NEAR_RANGE = 0.5  # [m] scale of the closest solid points, which are limited to this length
N_OTHER_GATES = 2  # Gates that are neither the target nor the next one
N_TRAVEL_POLES = 2  # Closest poles that are also observed relative to the direction of travel
MISS_RANGE = 0.4  # [m] miss distance at which a collision course saturates in the observation
OBS_DIM = 99
# Planned line
LINE_DS = 0.075  # [m] distance between the samples of the line
LINE_LENGTH = 360  # Samples of the line, enough for 27 m. Shorter lines repeat their last point
LINE_WINDOW = (5, 8)  # Samples (back, forward) searched by the projection on the line
LINE_AHEAD = (3, 6, 12, 18, 27)  # Samples ahead on the line that the policy observes
LINE_PIN = 3  # Samples around a gate that smoothing must not move, so gates are entered straight
LINE_SMOOTHING = 40  # Iterations of the smoothing of the line
LINE_RELAXATION = 8  # Iterations that push the smoothed line out of objects
LINE_REPAIR_WIDTH = 10.0  # Samples, width of the part of the line that follows a revealed gate
LINE_REPAIR_ITERATIONS = 4  # Iterations that push a repaired line out of objects
LINE_GATE_CLEARANCE = 0.3  # [m] distance from the line to gate solids that the relaxation aims for
LINE_POLE_CLEARANCE = 0.35  # [m] and to poles
TAKEOFF_HEIGHT = 0.3  # [m] rise before the line turns towards the first gate
GATE_EXIT = 0.45  # [m] the line continues this far along the gate normal behind a gate
GATE_APPROACH = 0.55  # [m] and starts this far in front of a gate
GATE_HALF_WIDTH = 0.75  # [m] half width of a gate for detours: outer frame, drone and a margin
GATE_TOP_CLEARANCE = 0.56  # [m] height above the centre of a gate at which the line may cross it
POLE_CLEARANCE = 0.35  # [m] a pole closer than this to the line causes a detour
DETOUR_HEIGHT_MIN = 0.25  # [m] lowest height of a detour point
N_DETOURS = 2  # Detours per segment between two gates


def gate_frame(xp: Any, state: dict[str, Any], gate_sequence: Any) -> tuple[Any, Any]:
    """Cosine and sine of the yaw of the target gate, which defines the frame of the policy."""
    n_passes = gate_sequence.shape[0]
    target = gate_sequence[xp.minimum(state["n_gates_passed"], n_passes - 1)]
    quat = state["gates_quat"][xp.arange(target.shape[0]), target]
    normal = quat_to_matrix(xp, quat)[..., 0]
    norm = xp.sqrt(normal[..., 0] ** 2 + normal[..., 1] ** 2) + 1e-9
    return normal[..., 0] / norm, normal[..., 1] / norm


def to_frame(xp: Any, vectors: Any, c: Any, s: Any) -> Any:
    """Rotate world-frame vectors (n, ..., 2 or 3) into the frame given by c and s (n,)."""
    shape = c.shape + (1,) * (vectors.ndim - c.ndim - 1)
    c, s = c.reshape(shape), s.reshape(shape)
    x = c * vectors[..., 0] + s * vectors[..., 1]
    y = -s * vectors[..., 0] + c * vectors[..., 1]
    if vectors.shape[-1] == 2:
        return xp.stack([x, y], axis=-1)
    return xp.stack([x, y, vectors[..., 2]], axis=-1)


def closest_solids(xp: Any, pos: Any, gates_pos: Any, gates_quat: Any) -> Any:
    """Vector from each drone to the closest point of the frame and stand of each gate.

    Args:
        xp: The array namespace.
        pos: The drone positions (n, 3).
        gates_pos: The gate positions (n, g, 3) or the same gates for all drones (g, 3).
        gates_quat: The gate orientations (n, g, 4) or (g, 4).

    Returns:
        The vectors in the world frame (n, g, 3).
    """
    rot = quat_to_matrix(xp, gates_quat)  # (n, g, 3, 3), columns are the gate axes
    local = xp.sum(rot * (pos[:, None] - gates_pos)[..., None], axis=-2)[:, :, None]  # (n, g, 1, 3)
    box_pos, box_half = xp.asarray(GATE_BOX_POS), xp.asarray(GATE_BOX_HALF)
    offset = xp.clip(local, box_pos - box_half, box_pos + box_half) - local  # (n, g, b, 3)
    closest = xp.argmin(xp.sum(offset**2, axis=-1), axis=2)  # (n, g)
    row = xp.arange(pos.shape[0])[:, None]
    column = xp.arange(gates_pos.shape[-2])[None]
    best = offset[row, column, closest]
    return xp.sum(rot * best[..., None, :], axis=-1)


def limit_length(xp: Any, vectors: Any, length: float) -> Any:
    """Scale vectors down to a maximum length, in units of that length."""
    size = xp.sqrt(xp.sum(vectors**2, axis=-1, keepdims=True) + 1e-12)
    return vectors / xp.maximum(size, length)


def along_and_across(xp: Any, vectors: Any, direction: Any) -> Any:
    """Horizontal components of vectors (n, ..., 2) along and across directions (n, 2)."""
    shape = (direction.shape[0],) + (1,) * (vectors.ndim - 2) + (2,)
    direction = direction.reshape(shape)
    along = xp.sum(vectors * direction, axis=-1)
    across = vectors[..., 0] * direction[..., 1] - vectors[..., 1] * direction[..., 0]
    return xp.stack([along, across], axis=-1)


def cross_2d(xp: Any, a: Any, b: Any) -> Any:
    """Scalar cross product of two-dimensional vectors."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def detour_points(
    xp: Any, start: Any, end: Any, gates_pos: Any, lateral: Any, poles: Any
) -> tuple[Any, Any]:
    """Points that lead a straight piece of line around the first objects that are in its way.

    Args:
        xp: The array namespace.
        start: The start of the piece of line (3,).
        end: The end of the piece of line (3,).
        gates_pos: The gate positions (g, 3).
        lateral: The horizontal lateral axes of the gates (g, 2).
        poles: The pole positions (o, 3).

    Returns:
        The detour points (N_DETOURS, 3) in the order along the line, which equal the end if there
        is nothing in the way.
    """
    a, d = start[:2], end[:2] - start[:2]
    length2 = xp.sum(d**2) + 1e-6
    side = xp.stack([-d[1], d[0]]) / xp.sqrt(length2)

    def path_length(point: Any) -> Any:
        return xp.sqrt(xp.sum((point - a) ** 2, axis=-1)) + xp.sqrt(
            xp.sum((point - end[:2]) ** 2, axis=-1)
        )

    # Poles: pass on the side with the shorter path, which is the side the line is on if it is open
    to_pole = poles[:, :2] - a
    t_pole = xp.clip(xp.sum(to_pole * d, axis=-1) / length2, 0.0, 1.0)
    gap = to_pole - t_pole[:, None] * d
    left = poles[:, :2] + side * (POLE_CLEARANCE + 0.05)
    right = poles[:, :2] - side * (POLE_CLEARANCE + 0.05)
    xy_pole = xp.where((path_length(left) <= path_length(right))[:, None], left, right)
    blocked_pole = xp.sum(gap**2, axis=-1) < POLE_CLEARANCE**2

    # Gates are walls that end at their half width: pass around the end that is closer
    wall = lateral * GATE_HALF_WIDTH
    to_gate = gates_pos[:, :2] - a
    denominator = cross_2d(xp, d[None], wall)
    safe = xp.where(xp.abs(denominator) > 1e-6, denominator, 1e-6)
    t_gate = cross_2d(xp, to_gate, wall) / safe
    u_gate = cross_2d(xp, to_gate, d[None]) / safe
    height = start[2] + t_gate * (end[2] - start[2])
    blocked_gate = (
        (xp.abs(denominator) > 1e-6)
        & (t_gate >= 0.0)
        & (t_gate <= 1.0)
        & (xp.abs(u_gate) <= 1.0)
        & (height < gates_pos[:, 2] + GATE_TOP_CLEARANCE)
    )
    margin = 1.0 + 0.05 / GATE_HALF_WIDTH
    plus, minus = gates_pos[:, :2] + wall * margin, gates_pos[:, :2] - wall * margin
    xy_gate = xp.where((path_length(plus) <= path_length(minus))[:, None], plus, minus)

    never = 10.0
    t_all = xp.concatenate(
        [xp.where(blocked_pole, t_pole, never), xp.where(blocked_gate, t_gate, never)]
    )
    xy_all = xp.concatenate([xy_pole, xy_gate])
    first = xp.argsort(t_all)[:N_DETOURS]
    t_first = xp.clip(t_all[first], 0.0, 1.0)
    z = xp.maximum(start[2] + t_first * (end[2] - start[2]), DETOUR_HEIGHT_MIN)
    points = xp.concatenate([xy_all[first], z[:, None]], axis=-1)
    return xp.where((t_all[first] < never)[:, None], points, end[None])


def axis_waypoint(
    xp: Any,
    gate: Any,
    normal: Any,
    lateral: Any,
    poles: Any,
    reach: float,
    direction: float,
    hint: Any,
) -> Any:
    """The point where the line leaves the axis of a gate, beside a pole that is on that axis.

    Args:
        xp: The array namespace.
        gate: The gate position (3,).
        normal: The horizontal gate normal (3,).
        lateral: The horizontal lateral axis of the gate (2,).
        poles: The pole positions (o, 3).
        reach: The distance along the axis of the point if there is no pole in the way.
        direction: 1 for the exit side of the gate, -1 for the entry side.
        hint: Where the line comes from or leads to, relative to the gate (2,). The line passes a
            pole on that side, because the other side would make it double back.

    Returns:
        The waypoint (3,).
    """
    rel = poles[:, :2] - gate[:2]
    along = direction * xp.sum(rel * normal[:2], axis=-1)
    across = xp.sum(rel * lateral, axis=-1)
    near = (along > 0.1) & (along < reach + POLE_CLEARANCE) & (xp.abs(across) < POLE_CLEARANCE)
    first = xp.argmin(xp.where(near, along, 1e3))
    hit = near.any()
    side = xp.where(xp.sum(hint * lateral) >= 0.0, 1.0, -1.0)
    distance = xp.where(hit, xp.minimum(along[first], reach), reach)
    offset = xp.where(hit, across[first] + side * (POLE_CLEARANCE + 0.05), 0.0)
    point = gate + direction * normal * distance
    return xp.concatenate([point[:2] + lateral * offset, point[2:]])


def smooth_line(xp: Any, samples: Any, pinned: Any) -> Any:
    """One iteration of smoothing of a line, which does not move the pinned samples."""
    before = xp.concatenate([samples[:1], samples[:-1]])
    after = xp.concatenate([samples[1:], samples[-1:]])
    return xp.where(pinned, samples, 0.5 * samples + 0.25 * (before + after))


def push_line_out(
    xp: Any, samples: Any, pinned: Any, gates_pos: Any, gates_quat: Any, poles: Any
) -> Any:
    """Move the samples of a line that are too close to a gate solid or a pole away from it."""
    solids = closest_solids(xp, samples, gates_pos, gates_quat)
    nearest = xp.argmin(xp.sum(solids**2, axis=-1), axis=1)
    solid = solids[xp.arange(LINE_LENGTH), nearest]
    size = xp.sqrt(xp.sum(solid**2, axis=-1, keepdims=True) + 1e-9)
    push = -solid / size * xp.maximum(LINE_GATE_CLEARANCE - size, 0.0)
    away = samples[:, None, :2] - poles[None, :, :2]
    size = xp.sqrt(xp.sum(away**2, axis=-1, keepdims=True) + 1e-9)
    pole_push = xp.sum(away / size * xp.maximum(LINE_POLE_CLEARANCE - size, 0.0), axis=1)
    zero = xp.zeros_like(pole_push[:, :1])
    moved = samples + push + xp.concatenate([pole_push, zero], axis=-1)
    return xp.where(pinned, samples, moved)


def pinned_samples(xp: Any, gate_samples: Any, last: Any) -> Any:
    """Mask of the samples that stay in place: around gate passes, the start and after the end."""
    index = xp.arange(LINE_LENGTH)
    near_gate = xp.abs(index[:, None] - gate_samples[None]) <= LINE_PIN
    return (near_gate.any(axis=1) | (index == 0) | (index >= last))[:, None]


def repair_line(
    xp: Any,
    line: Any,
    gate_index: Any,
    sequence: Any,
    nominal_pos: Any,
    nominal_quat: Any,
    gates_pos: Any,
    gates_quat: Any,
    poles: Any,
) -> Any:
    """Adapt a line planned on the nominal track to the poses that have been observed so far.

    The parts of the line around each gate pass follow the shift and rotation of that gate, then the
    line is pushed out of the objects. This keeps the sample indices, so that the progress along the
    line does not jump. The result only depends on the plan and the current observation.

    Args:
        xp: The array namespace.
        line: The line planned on the nominal track (LINE_LENGTH, 3).
        gate_index: The sample index of each gate pass (m,).
        sequence: The gate IDs of the gate order (m,).
        nominal_pos: The nominal gate positions (g, 3).
        nominal_quat: The nominal gate orientations (g, 4).
        gates_pos: The observed gate positions (g, 3).
        gates_quat: The observed gate orientations (g, 4).
        poles: The observed pole positions (o, 3).

    Returns:
        The repaired line (LINE_LENGTH, 3).
    """

    def horizontal_axis(quat: Any) -> Any:
        axis = quat_to_matrix(xp, quat)[:, :2, 0]
        return axis / xp.sqrt(xp.sum(axis**2, axis=-1, keepdims=True) + 1e-12)

    before, after = horizontal_axis(nominal_quat)[sequence], horizontal_axis(gates_quat)[sequence]
    cosine = xp.sum(before * after, axis=-1)[None]  # (1, m)
    sine = cross_2d(xp, before, after)[None]
    rel = line[:, None] - nominal_pos[sequence][None]  # (l, m, 3)
    turn_x = (cosine - 1.0) * rel[..., 0] - sine * rel[..., 1]
    turn_y = sine * rel[..., 0] + (cosine - 1.0) * rel[..., 1]
    turn = xp.stack([turn_x, turn_y, xp.zeros_like(turn_x)], axis=-1)
    shift = (gates_pos - nominal_pos)[sequence][None]
    index = xp.arange(LINE_LENGTH)[:, None]
    weight = xp.exp(-(((index - gate_index[None]) / LINE_REPAIR_WIDTH) ** 2))[..., None]
    moved = line + xp.sum(weight * (shift + turn), axis=1)

    pinned = pinned_samples(xp, gate_index, gate_index[-1] + GATE_EXIT / LINE_DS)
    for _ in range(LINE_REPAIR_ITERATIONS):
        pushed = push_line_out(xp, moved, pinned, gates_pos, gates_quat, poles)
        moved = smooth_line(xp, smooth_line(xp, pushed, pinned), pinned)
    return moved


def plan_line(
    xp: Any, start: Any, gates_pos: Any, gates_quat: Any, poles: Any, sequence: Any
) -> tuple[Any, Any]:
    """Plan a line through the gates of a track, sampled at a constant distance.

    The line leaves a gate along its normal and enters the next one along its normal. In between it
    passes around the first objects in its way, including the gate that was just passed and the
    target gate if that has to be entered from behind. The corners are then smoothed.

    Args:
        xp: The array namespace.
        start: The start position (3,).
        gates_pos: The nominal gate positions (g, 3).
        gates_quat: The nominal gate orientations (g, 4).
        poles: The nominal pole positions (o, 3).
        sequence: The gate IDs of the gate order (m,).

    Returns:
        The line samples (LINE_LENGTH, 3) and the sample index of each gate pass (m,).
    """
    axis = quat_to_matrix(xp, gates_quat)[:, :2, 0]
    axis = axis / xp.sqrt(xp.sum(axis**2, axis=-1, keepdims=True) + 1e-12)
    zeros = xp.zeros_like(axis[:, :1])
    normals = xp.concatenate([axis, zeros], axis=-1)
    lateral = xp.stack([-axis[:, 1], axis[:, 0]], axis=-1)

    waypoints, previous = [], start
    leave = start + xp.asarray([0.0, 0.0, TAKEOFF_HEIGHT], dtype=gates_pos.dtype)
    n_passes = sequence.shape[0]
    for k in range(n_passes):
        target = sequence[k]
        gate = gates_pos[target]
        enter = axis_waypoint(
            xp,
            gate,
            normals[target],
            lateral[target],
            poles,
            GATE_APPROACH,
            -1.0,
            leave[:2] - gate[:2],
        )
        detours = detour_points(xp, leave, enter, gates_pos, lateral, poles)
        waypoints += [previous[None], leave[None], detours, enter[None]]
        previous = gate
        onward = gates_pos[sequence[min(k + 1, n_passes - 1)]][:2] - gate[:2]
        leave = axis_waypoint(
            xp, gate, normals[target], lateral[target], poles, GATE_EXIT, 1.0, onward
        )
    waypoints += [previous[None], leave[None]]
    points = xp.concatenate(waypoints)

    def resample(line: Any, marks: Any) -> tuple[Any, Any]:
        """Sample a polyline at a constant distance, and convert sample marks along with it."""
        steps = xp.sqrt(xp.sum((line[1:] - line[:-1]) ** 2, axis=-1)) + 1e-5
        length = xp.concatenate([xp.zeros_like(steps[:1]), xp.cumsum(steps)])
        grid = xp.arange(LINE_LENGTH) * LINE_DS
        axes = [xp.interp(grid, length, line[:, i]) for i in range(3)]
        return xp.stack(axes, axis=-1), length[marks] / LINE_DS

    n_per_pass = 3 + N_DETOURS
    marks = n_per_pass * xp.arange(1, sequence.shape[0] + 1)
    line, gate_samples = resample(points, marks)
    # Number of samples on the polyline, which ends after the last gate
    last = gate_samples[-1] + GATE_EXIT / LINE_DS

    pinned = pinned_samples(xp, gate_samples, last)
    for _ in range(LINE_SMOOTHING):
        line = smooth_line(xp, line, pinned)
    for _ in range(LINE_RELAXATION):
        pushed = push_line_out(xp, line, pinned, gates_pos, gates_quat, poles)
        line = smooth_line(xp, smooth_line(xp, pushed, pinned), pinned)

    # Smoothing shortens corners, so the samples and the gate indices are made regular again
    steps = xp.sqrt(xp.sum((line[1:] - line[:-1]) ** 2, axis=-1)) + 1e-5
    length = xp.concatenate([xp.zeros_like(steps[:1]), xp.cumsum(steps)])
    gate_length = xp.interp(gate_samples, xp.arange(LINE_LENGTH) * 1.0, length)
    grid = xp.arange(LINE_LENGTH) * LINE_DS
    line = xp.stack([xp.interp(grid, length, line[:, i]) for i in range(3)], axis=-1)
    gate_index = xp.clip(xp.round(gate_length / LINE_DS), 0, LINE_LENGTH - 1).astype(xp.int32)
    return line, gate_index


def project_on_line(
    xp: Any, lines: Any, gate_index: Any, track: Any, line_idx: Any, pos: Any, passed: Any
) -> tuple[Any, Any, Any]:
    """Project positions on the part of a line that is reachable before the next gate.

    Args:
        xp: The array namespace.
        lines: The lines (t, LINE_LENGTH, 3).
        gate_index: The sample index of each gate pass of the lines (t, m).
        track: The line of each drone (n,).
        line_idx: The current sample index of each drone (n,).
        pos: The drone positions (n, 3).
        passed: The number of gate passes (n,).

    Returns:
        The sample index, the refined progress in metres and the distance to the line, each (n,).
    """
    row = xp.arange(pos.shape[0])
    n_passes, last = gate_index.shape[1], LINE_LENGTH - 1
    limit = gate_index[track, xp.minimum(passed, n_passes - 1)]
    limit = xp.where(passed < n_passes, limit, last)
    offsets = xp.arange(-LINE_WINDOW[0], LINE_WINDOW[1] + 1)
    candidates = xp.clip(line_idx[:, None] + offsets[None], 0, limit[:, None])
    distance2 = xp.sum((pos[:, None] - lines[track[:, None], candidates]) ** 2, axis=-1)
    best = xp.argmin(distance2, axis=1)
    index = candidates[row, best]
    # Refine within the sample so that the progress does not jump with the resolution
    point = lines[track, index]
    tangent = lines[track, xp.minimum(index + 1, last)] - point
    size = xp.sqrt(xp.sum(tangent**2, axis=-1)) + 1e-6
    along = xp.sum((pos - point) * tangent, axis=-1) / size
    progress = xp.minimum(index * LINE_DS + xp.clip(along, -LINE_DS, LINE_DS), limit * LINE_DS)
    return index, progress, xp.sqrt(distance2[row, best])


def line_ahead(xp: Any, lines: Any, track: Any, line_idx: Any) -> Any:
    """Points ahead on the lines of the drones (n, len(LINE_AHEAD), 3)."""
    offsets = xp.asarray(LINE_AHEAD)[None]
    return lines[track[:, None], xp.minimum(line_idx[:, None] + offsets, LINE_LENGTH - 1)]


def collision_course(xp: Any, vectors: Any, velocity: Any) -> Any:
    """How a straight flight would pass objects, at the resolution that decides hit or miss.

    Args:
        xp: The array namespace.
        vectors: Horizontal vectors from the drones to the objects (n, ..., 2) [m].
        velocity: Horizontal velocities of the drones (n, 2) [m/s].

    Returns:
        The signed miss distance, saturating at MISS_RANGE, and the time to the closest approach in
        units of 2 s (n, ..., 2).
    """
    speed = xp.sqrt(xp.sum(velocity**2, axis=-1, keepdims=True) + 0.01)
    course = along_and_across(xp, vectors, velocity / speed)
    speed = speed.reshape((speed.shape[0],) + (1,) * (vectors.ndim - 2))
    miss = xp.clip(course[..., 1] / MISS_RANGE, -1.0, 1.0)
    time = xp.clip(course[..., 0] / (speed + 0.2), -0.5, 2.0) / 2.0
    return xp.stack([miss, time], axis=-1)


def general_observation(
    xp: Any,
    state: dict[str, Any],
    gate_sequence: Any,
    act_prev: Any,
    act_prev2: Any,
    ratio: Any,
    speed_cap: Any,
    arena: tuple[Any, Any],
    ahead: Any,
) -> Any:
    """Assemble the policy input for a batch of drones.

    Args:
        xp: The array namespace.
        state: Batched race observation with the keys of the environment's observation space.
        gate_sequence: The gate ID of each entry of the gate order (m,).
        act_prev: The last action in the world frame (n, 3).
        act_prev2: The action before that in the world frame (n, 3).
        ratio: The estimated relative mass (n,).
        speed_cap: The speed cap [m/s] (n,).
        arena: Lower and upper arena limits (2,).
        ahead: Points ahead on the planned line (n, len(LINE_AHEAD), 3).

    Returns:
        The policy input (n, OBS_DIM).
    """
    pos, vel = state["pos"], state["vel"]
    n, n_passes = pos.shape[0], gate_sequence.shape[0]
    row = xp.arange(n)
    c, s = gate_frame(xp, state, gate_sequence)
    rot = quat_to_matrix(xp, state["quat"])
    ang_vel_world = xp.sum(rot * state["ang_vel"][:, None, :], axis=-1)

    passed = state["n_gates_passed"]
    pass_now = xp.minimum(passed, n_passes - 1)
    pass_next = xp.minimum(passed + 1, n_passes - 1)
    target, following = gate_sequence[pass_now], gate_sequence[pass_next]
    visited = xp.float32(1.0) * state["gates_visited"]

    gates_rel = state["gates_pos"] - pos[:, None]
    normals = quat_to_matrix(xp, state["gates_quat"])[..., 0]
    # The remaining gates are obstacles on the way, the closest ones matter
    index = xp.arange(gates_rel.shape[1])[None]
    excluded = (index == target[:, None]) | (index == following[:, None])
    distance = xp.sum(gates_rel**2, axis=-1) + 1e3 * excluded
    order = xp.argsort(distance, axis=1)[:, :N_OTHER_GATES]
    others_rel = gates_rel[row[:, None], order]
    others_normal = normals[row[:, None], order]

    poles_rel = state["obstacles_pos"][..., :2] - pos[:, None, :2]
    pole_order = xp.argsort(xp.sum(poles_rel**2, axis=-1), axis=1)
    poles_rel = poles_rel[row[:, None], pole_order]
    poles_seen = xp.float32(1.0) * state["obstacles_visited"][row[:, None], pole_order]

    # Arena limits appear as a signed proximity that is zero far away from them
    low, high = arena
    wall = xp.clip(1.0 - (high[None] - pos[:, :2]) / WALL_RANGE, 0.0, 1.0)
    wall = wall - xp.clip(1.0 - (pos[:, :2] - low[None]) / WALL_RANGE, 0.0, 1.0)
    zero = xp.zeros_like(wall[:, 0])
    wall_x = xp.stack([wall[:, 0], zero], axis=-1)
    wall_y = xp.stack([zero, wall[:, 1]], axis=-1)

    # Closest solid point of the target gate and of any other gate
    solids = closest_solids(xp, pos, state["gates_pos"], state["gates_quat"])
    other = xp.argmin(xp.sum(solids**2, axis=-1) + 1e3 * (index == target[:, None]), axis=1)
    solid_target = to_frame(xp, limit_length(xp, solids[row, target], NEAR_RANGE), c, s)
    solid_other = to_frame(xp, limit_length(xp, solids[row, other], NEAR_RANGE), c, s)

    # Where the nearby objects are relative to the direction of travel, zero when not moving
    vel_frame = to_frame(xp, vel[:, :2], c, s)
    travel = vel_frame / xp.sqrt(xp.sum(vel_frame**2, axis=-1, keepdims=True) + 0.01)
    poles_frame = to_frame(xp, poles_rel[:, :N_TRAVEL_POLES], c, s) / POS_SCALE

    terms = [
        to_frame(xp, rot[..., 2], c, s),
        to_frame(xp, vel, c, s) / VEL_SCALE,
        to_frame(xp, ang_vel_world, c, s) / ANG_VEL_SCALE,
        pos[:, 2:] / POS_SCALE,
        to_frame(xp, act_prev[:, :2], c, s),
        act_prev[:, 2:],
        to_frame(xp, act_prev2[:, :2], c, s),
        act_prev2[:, 2:],
        ((ratio - 1.0) / MASS_RATIO_SCALE)[:, None],
        (speed_cap / SPEED_SCALE)[:, None],
        to_frame(xp, gates_rel[row, target], c, s) / POS_SCALE,
        to_frame(xp, normals[row, target], c, s),
        visited[row, target][:, None],
        to_frame(xp, gates_rel[row, following], c, s) / POS_SCALE,
        to_frame(xp, normals[row, following], c, s),
        visited[row, following][:, None],
        xp.float32(1.0) * (pass_next == pass_now)[:, None],
        ((n_passes - passed) / n_passes)[:, None],
        to_frame(xp, others_rel, c, s).reshape(n, -1) / POS_SCALE,
        to_frame(xp, others_normal, c, s).reshape(n, -1),
        visited[row[:, None], order],
        to_frame(xp, poles_rel, c, s).reshape(n, -1) / POS_SCALE,
        poles_seen,
        to_frame(xp, wall_x, c, s),
        to_frame(xp, wall_y, c, s),
        solid_target,
        solid_other,
        along_and_across(xp, poles_frame, travel).reshape(n, -1),
        along_and_across(xp, solid_target[:, :2], travel),
        along_and_across(xp, solid_other[:, :2], travel),
        to_frame(xp, ahead - pos[:, None], c, s).reshape(n, -1) / POS_SCALE,
        collision_course(xp, poles_frame * POS_SCALE, vel_frame).reshape(n, -1),
        collision_course(xp, solid_other[:, :2] * NEAR_RANGE, vel_frame),
    ]
    return xp.concatenate(terms, axis=-1)


def general_command(
    xp: Any, action: Any, c: Any, s: Any, limits: dict[str, Any]
) -> tuple[Any, Any]:
    """Convert policy actions to attitude commands.

    The first two action entries are the desired tilt of the thrust axis in the frame of the target
    gate, the third one the collective thrust. Yaw is held at zero, the tilt does not depend on it.

    Args:
        xp: The array namespace.
        action: Policy actions in [-1, 1]^3 (n, 3).
        c: Cosine of the yaw of the frame (n,).
        s: Sine of the yaw of the frame (n,).
        limits: The action limits.

    Returns:
        The commands [roll, pitch, yaw, thrust] (n, 4) and the actions in the world frame (n, 3).
    """
    a = xp.clip(action, -1.0, 1.0)
    world_x = c * a[..., 0] - s * a[..., 1]
    world_y = s * a[..., 0] + c * a[..., 1]
    # Limit the magnitude of the tilt, so that its limit does not depend on the direction
    size = xp.sqrt(world_x**2 + world_y**2 + 1e-12)
    factor = xp.minimum(1.0, 1.0 / size)
    world_x, world_y = world_x * factor, world_y * factor
    tilt = xp.minimum(size, 1.0) * limits["max_tilt"]
    # Thrust axis z = (sin(p) cos(r), -sin(r), cos(p) cos(r)) for roll r, pitch p and yaw zero
    sinc = xp.sinc(tilt / np.pi) * limits["max_tilt"]
    z_x, z_y, z_z = sinc * world_x, sinc * world_y, xp.cos(tilt)
    roll = -xp.arcsin(xp.clip(z_y, -1.0, 1.0))
    pitch = xp.arctan2(z_x, z_z)
    thrust = limits["thrust_mid"] + a[..., 2] * limits["thrust_half"]
    command = xp.stack([roll, pitch, xp.zeros_like(thrust), thrust], axis=-1)
    return command, xp.stack([world_x, world_y, a[..., 2]], axis=-1)


class RLGeneral(Controller):
    """Race controller running a trained track-independent policy."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load the policy.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        ckpt = np.load(POLICY_PATH)
        self._layers = [(ckpt[f"w{i}"], ckpt[f"b{i}"]) for i in range(int(ckpt["n_layers"]))]
        self._limits = {k: float(ckpt[k]) for k in ("thrust_mid", "thrust_half", "max_tilt")}
        self._mass_nominal = float(ckpt["mass_nominal"])
        self._arena = (ckpt["arena_low"], ckpt["arena_high"])
        self._dt = 1.0 / config.env.freq
        self._gate_sequence = np.asarray(obs["gate_sequence"], dtype=int)
        self._n_passes = self._gate_sequence.shape[0]
        self.speed_cap = np.full(1, SPEED_CAP, dtype=np.float32)
        self._nominal_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        self._nominal_quat = np.asarray(obs["gates_quat"], dtype=np.float32)
        self._plan, gate_index = plan_line(
            np,
            np.asarray(obs["pos"], dtype=np.float32),
            self._nominal_pos,
            self._nominal_quat,
            np.asarray(obs["obstacles_pos"], dtype=np.float32),
            self._gate_sequence,
        )
        self._gate_index = gate_index[None]
        self.episode_reset()

    def episode_reset(self):
        """Reset the action history and the mass estimate."""
        self._act_prev = np.zeros((1, 3), dtype=np.float32)
        self._act_prev2 = np.zeros((1, 3), dtype=np.float32)
        self._est = mass_estimate_init(np, 1, self._mass_nominal)
        self._vel_prev = None
        self._thrust_cmd = np.full((1,), self._mass_nominal * G, dtype=np.float32)
        self._line_idx = np.zeros(1, dtype=int)
        self._lines = self._plan[None]
        self._revealed = None

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
        revealed = np.concatenate([obs["gates_visited"], obs["obstacles_visited"]])
        if self._revealed is None or np.any(revealed != self._revealed):
            self._revealed = revealed
            line = repair_line(
                np,
                self._plan,
                self._gate_index[0],
                self._gate_sequence,
                self._nominal_pos,
                self._nominal_quat,
                np.asarray(obs["gates_pos"]),
                np.asarray(obs["gates_quat"]),
                np.asarray(obs["obstacles_pos"]),
            )
            self._lines = line[None]
        track = np.zeros(1, dtype=int)
        self._line_idx, _, _ = project_on_line(
            np,
            self._lines,
            self._gate_index,
            track,
            self._line_idx,
            state["pos"],
            state["n_gates_passed"],
        )
        policy_in = general_observation(
            np,
            state,
            self._gate_sequence,
            self._act_prev,
            self._act_prev2,
            mass_ratio(np, self._est, self._mass_nominal),
            self.speed_cap,
            self._arena,
            line_ahead(np, self._lines, track, self._line_idx),
        )
        action = mlp_forward(np, self._layers, policy_in.astype(np.float32))
        c, s = gate_frame(np, state, self._gate_sequence)
        command, world_action = general_command(np, action, c, s, self._limits)
        self._act_prev2, self._act_prev = self._act_prev, world_action
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
