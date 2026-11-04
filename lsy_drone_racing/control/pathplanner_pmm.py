"""Point-mass-model (PMM) minimum-time trajectory planner and racing controller.

Planner (after the CTU-MRS ``pmm_uav_planner`` [1], building on [2]):

1. **Point mass model.** The drone is a point mass with bounded thrust. On every axis the motion
   between two waypoints is the bang-bang solution of a double integrator (full acceleration in
   one direction, then in the other). The minimum time of one axis has a closed form.
2. **Axis synchronisation.** All axes must arrive at the next waypoint at the same time. The
   slowest axis defines the segment time; the others are re-solved for exactly that time with a
   reduced acceleration (closed form). Axes whose reachable times have a gap jump to the next
   feasible time.
3. **Thrust decomposition.** The real limit is the collective thrust ``|a + g| <= F_MAX``, not a
   box per axis. The per-axis limits of every segment are iteratively redistributed towards the
   direction the segment needs, so the full thrust can be used.
4. **Waypoint velocity optimisation.** The waypoint velocities are optimised with a projected
   gradient method using the analytic gradient of the bang-bang segment times, alternating with
   the thrust decomposition. At gates the velocity is restricted to a cone around the gate normal.

Controller (full-state interface of the onboard Mellinger controller):

* Position, velocity and acceleration of the PMM reference are commanded. The acceleration
  feed-forward is taken ``FF_LOOKAHEAD`` ahead and averaged over ``ACC_SMOOTH_W`` (the bang-bang
  steps cannot be followed by the attitude loop), and the thrust scaling of the onboard
  controller is inverted.
* Waypoints: gates in the configured order plus level-2 specific helper points (obstacle avoidance
  and support points). Gates whose true pose is still unknown are approached straight and slowly
  (approach point, speed cap, slow zone), so the pose error revealed at 0.7 m can be corrected.
* Re-planning: when an observed gate or obstacle pose changes, the remaining trajectory is
  re-planned from the reference state ``T_COMMIT`` ahead, warm-started with the previous waypoint
  velocities. Nearly reached or passed helper points are skipped, and plans that come too close to
  a pole or gate frame (collision box of the drone) are repaired with detour points.

[1] K. Teissing, M. Novosad, R. Penicka, M. Saska, "Real-time Planning of Minimum-time Trajectories
    for Agile UAV Flight", IEEE RA-L, 2024. https://github.com/ctu-mrs/pmm_uav_planner
[2] R. Penicka, D. Scaramuzza, "Minimum-Time Quadrotor Waypoint Flight in Cluttered Environments",
    IEEE RA-L, 2022.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.control.core import load_fn_params
from crazyflow.control.mellinger import state2attitude
from crazyflow.dynamics import load_params as load_dynamics_params
from crazyflow.sim.visualize import draw_line, draw_points
from lsy_drone_racing.control import Controller
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------------------------
# Planner parameters (module level so that a test harness can override them)
# ---------------------------------------------------------------------------------------------
G = 9.81  # [m/s^2]
F_MAX = 13.0  # [m/s^2] mass-normalised collective thrust |a + g| used for planning (hw ~18.4)
A_Z_MAX = 4.0  # [m/s^2] vertical acceleration bound (up and down, keeps thrust >= g - A_Z_MAX)
A_Z_MIN = 0.5  # [m/s^2] lower bound of the vertical acceleration limit in the decomposition
A_H_MIN_FRAC = 0.05  # minimum share of the horizontal budget per horizontal axis
V_MAX = 3.0  # [m/s] bound on the waypoint speed
V_GATE_MIN = 0.3  # [m/s] minimum speed through a gate (along its normal)
GATE_CONE = np.deg2rad(35.0)  # [rad] max. angle between the velocity and the gate normal
V_INIT = 1.5  # [m/s] initial guess of the waypoint speed
N_DECOMP = 3  # thrust decomposition iterations per outer iteration
N_OUTER = 3  # alternations between thrust decomposition and velocity descent
OPT_ITERS = 120  # max. gradient iterations of the waypoint velocity optimisation (cold start)
OPT_ITERS_WARM = 40  # max. gradient iterations when re-planning (warm start)
OPT_STEP0 = 2.0  # initial gradient step [(m/s)^2 / s]

_EPS = 1e-9


# ---------------------------------------------------------------------------------------------
# 1D bang-bang building blocks (vectorised over arbitrary array shapes)
# ---------------------------------------------------------------------------------------------
def pmm_candidates(
    dp: NDArray, v0: NDArray, v1: NDArray, a: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """All bang-bang candidates of the 1D problem (p0, v0) -> (p0 + dp, v1) with |acc| <= a.

    A bang-bang profile accelerates with +a (or -a) until the switching velocity vs is reached and
    then with -a (or +a) until v1. Energy balance gives vs^2 = (v0^2 + v1^2)/2 +- a dp, and each
    sign of the root yields a candidate. Invalid candidates get the time ``inf``.

    Args:
        dp: Displacement.
        v0: Initial velocity.
        v1: Final velocity.
        a: Acceleration bound (> 0).

    Returns:
        Times (4, ...) and their gradients w.r.t. v0 and v1 (4, ...). Candidate order: (+a,-a)
        with vs < 0, (+a,-a) with vs > 0, (-a,+a) with vs > 0, (-a,+a) with vs < 0.
    """
    a = np.maximum(a, 1e-6)
    m = 0.5 * (v0 * v0 + v1 * v1)
    s_up, s_dn = m + a * dp, m - a * dp
    r_up, r_dn = np.sqrt(np.maximum(s_up, 0.0)), np.sqrt(np.maximum(s_dn, 0.0))
    vmax, vmin = np.maximum(v0, v1), np.minimum(v0, v1)
    tol = 1e-9 + 1e-9 * np.abs(vmax)
    vs = np.stack([-r_up, r_up, r_dn, -r_dn])
    sign = np.array([1.0, 1.0, -1.0, -1.0]).reshape((4,) + (1,) * np.ndim(dp))
    ok = np.stack(
        [
            (s_up >= -tol) & (-r_up >= vmax - tol),
            (s_up >= -tol) & (r_up >= vmax - tol),
            (s_dn >= -tol) & (r_dn <= vmin + tol),
            (s_dn >= -tol) & (-r_dn <= vmin + tol),
        ]
    )
    t = np.maximum(sign * (2.0 * vs - v0 - v1) / a, 0.0)
    t = np.where(ok, t, np.inf)
    vs_safe = np.where(np.abs(vs) < 1e-6, np.where(vs < 0, -1e-6, 1e-6), vs)
    g0 = sign * (v0 / vs_safe - 1.0) / a
    g1 = sign * (v1 / vs_safe - 1.0) / a
    return t, g0, g1


def pmm_sync(dp: NDArray, v0: NDArray, v1: NDArray, t: NDArray) -> tuple[NDArray, NDArray]:
    """Bang-bang profile that needs exactly the time t (axis synchronisation).

    With the signed acceleration a of the first phase, t1 + t2 = t and a (t1 - t2) = v1 - v0, the
    displacement condition becomes  t^2 a^2 + 2 (t (v0 + v1) - 2 dp) a - (v1 - v0)^2 = 0.
    The roots have opposite signs; only the one with the larger magnitude gives t1, t2 >= 0.

    Returns:
        Signed acceleration of the first phase and the switching time t1.
    """
    t = np.maximum(t, _EPS)
    dv = v1 - v0
    b = 2.0 * (t * (v0 + v1) - 2.0 * dp)
    disc = np.sqrt(b * b + 4.0 * t * t * dv * dv)
    a = -(b + np.where(b >= 0.0, 1.0, -1.0) * disc) / (2.0 * t * t)
    a_safe = np.where(np.abs(a) < 1e-12, 1.0, a)
    t1 = np.where(np.abs(a) < 1e-12, 0.5 * t, 0.5 * (t + dv / a_safe))
    return a, np.clip(t1, 0.0, t)


@dataclass
class Segments:
    """Synchronised bang-bang segments between consecutive waypoints."""

    p0: NDArray  # (n, 3) start positions
    v0: NDArray  # (n, 3) start velocities
    acc: NDArray  # (n, 3) signed acceleration of the first phase (second phase: -acc)
    t1: NDArray  # (n, 3) switching times
    dt: NDArray  # (n,) segment durations
    grad0: NDArray  # (n, 3) d dt / d v0
    grad1: NDArray  # (n, 3) d dt / d v1
    bounds: NDArray  # (n, 3) per-axis acceleration bounds after thrust decomposition


def solve_segments(p: NDArray, v: NDArray, bounds: NDArray) -> Segments:
    """Minimum-time synchronised bang-bang segments for fixed waypoint velocities and bounds.

    Args:
        p: Waypoints (n+1, 3).
        v: Waypoint velocities (n+1, 3).
        bounds: Per-segment, per-axis acceleration bounds (n, 3).
    """
    p0, p1, v0, v1 = p[:-1], p[1:], v[:-1], v[1:]
    dp = p1 - p0
    n = len(dp)
    rows = np.arange(n)
    tc, g0c, g1c = pmm_candidates(dp, v0, v1, bounds)  # (4, n, 3)
    t_axis = tc.min(axis=0)  # (n, 3) min time per axis
    k = t_axis.argmax(axis=1)  # limiting axis
    c = tc[:, rows, k].argmin(axis=0)  # limiting candidate
    t = t_axis[rows, k].copy()
    acc, t1 = pmm_sync(dp, v0, v1, t[:, None])
    for _ in range(4):  # axes whose reachable times have a gap at t: jump to the next candidate
        bad = np.abs(acc) > bounds * (1.0 + 1e-6) + 1e-9
        if not bad.any():
            break
        later = np.where(tc > t[None, :, None] * (1.0 + 1e-9) + 1e-12, tc, np.inf)
        nxt = np.where(bad, later.min(axis=0), -np.inf)  # (n, 3)
        k_new = nxt.argmax(axis=1)
        t_new = nxt[rows, k_new]
        upd = bad.any(axis=1) & np.isfinite(t_new)
        t = np.where(upd, t_new, t)
        k = np.where(upd, k_new, k)
        c = np.where(upd, later[:, rows, k_new].argmin(axis=0), c)
        acc, t1 = pmm_sync(dp, v0, v1, t[:, None])
    grad0, grad1 = np.zeros_like(dp), np.zeros_like(dp)
    grad0[rows, k] = g0c[c, rows, k]
    grad1[rows, k] = g1c[c, rows, k]
    return Segments(p0, v0, acc, t1, t, grad0, grad1, bounds)


def initial_bounds(n: int, f_max: float) -> NDArray:
    """Per-axis bounds before the thrust decomposition: moderate vertical, rest horizontal."""
    a_z = min(2.0, A_Z_MAX)
    a_h = np.sqrt(max(f_max**2 - (G + a_z) ** 2, 1e-6) / 2.0)
    return np.tile([a_h, a_h, a_z], (n, 1))


def decompose_thrust(used: NDArray, f_max: float) -> NDArray:
    """Redistribute the thrust budget |a + g| <= f_max towards the acceleration a segment uses.

    The thrust vector needed in the worst corner of a segment is f = (|ax|, |ay|, g + |az|). It is
    scaled onto the thrust sphere, which gives the vertical bound; the remaining horizontal budget
    is split between x and y in proportion to their use.

    Args:
        used: Per-segment acceleration magnitudes (n, 3) of the synchronised solution.
        f_max: Mass-normalised thrust limit.
    """
    f = np.column_stack([used[:, 0], used[:, 1], G + used[:, 2]])
    lam = f_max / np.linalg.norm(f, axis=1)
    a_z = np.clip(lam * f[:, 2] - G, A_Z_MIN, A_Z_MAX)
    h_budget = np.sqrt(np.maximum(f_max**2 - (G + a_z) ** 2, 1e-6))
    h = used[:, :2]
    hn = np.linalg.norm(h, axis=1, keepdims=True)
    share = np.where(hn > 1e-6, h / np.maximum(hn, 1e-12), np.sqrt(0.5))
    share = np.maximum(share, A_H_MIN_FRAC)
    share /= np.linalg.norm(share, axis=1, keepdims=True)
    return np.column_stack([h_budget[:, None] * share, a_z])


def refine_bounds(p: NDArray, v: NDArray, bounds: NDArray, f_max: float, iters: int) -> NDArray:
    """Iterative thrust decomposition for fixed waypoint velocities.

    Every iterate lies on the thrust sphere, so any bounds returned are admissible; the iteration
    only moves the per-axis limits towards the acceleration direction each segment needs.
    """
    for _ in range(iters):
        seg = solve_segments(p, v, bounds)
        new = 0.5 * bounds + 0.5 * decompose_thrust(np.abs(seg.acc), f_max)
        # Keep the old bounds of segments that would get slower (the fixed point is not monotone)
        better = solve_segments(p, v, new).dt <= seg.dt + 1e-9
        bounds = np.where(better[:, None], new, bounds)
    return bounds


# ---------------------------------------------------------------------------------------------
# Waypoint velocity optimisation
# ---------------------------------------------------------------------------------------------
def _project(v: NDArray, dirs: NDArray, fixed: NDArray, vmax: NDArray) -> NDArray:
    """Project waypoint velocities onto their admissible sets.

    At gates the velocity must lie in a cone of half angle GATE_CONE around the gate normal (gate
    frames are thin, so moderately oblique crossings are safe) with a minimum normal speed. At all
    other waypoints only the speed is bounded.
    """
    v = v.copy()
    gate = ~np.isnan(dirs[:, 0]) & ~fixed
    if gate.any():
        n, vg = dirs[gate], v[gate]
        s = np.sum(vg * n, axis=1)
        w = vg - s[:, None] * n
        wn = np.linalg.norm(w, axis=1)
        tan_c, cos_c, sin_c = np.tan(GATE_CONE), np.cos(GATE_CONE), np.sin(GATE_CONE)
        outside = wn > tan_c * s
        # Project onto the cone boundary (second-order cone projection, polar part -> 0)
        u = np.where(wn[:, None] > 1e-9, w / np.maximum(wn, 1e-9)[:, None], 0.0)
        d = cos_c * n + sin_c * u
        proj = np.maximum(np.sum(vg * d, axis=1), 0.0)[:, None] * d
        vg = np.where(outside[:, None], proj, vg)
        s = np.sum(vg * n, axis=1)
        vg = vg + np.maximum(V_GATE_MIN - s, 0.0)[:, None] * n
        sp = np.linalg.norm(vg, axis=1)
        vg *= np.minimum(1.0, vmax[gate] / np.maximum(sp, 1e-9))[:, None]
        v[gate] = vg
    free = np.isnan(dirs[:, 0]) & ~fixed
    if free.any():
        n = np.linalg.norm(v[free], axis=1)
        v[free] *= np.minimum(1.0, vmax[free] / np.maximum(n, 1e-9))[:, None]
    return v


def _initial_velocities(p: NDArray, dirs: NDArray, speed: float = V_INIT) -> NDArray:
    """Heuristic initial guess: bisector direction of in- and outgoing waypoint legs."""
    v = np.zeros_like(p)
    for i in range(1, len(p)):
        d_in = p[i] - p[i - 1]
        d_out = p[i + 1] - p[i] if i + 1 < len(p) else d_in
        d = d_in / (np.linalg.norm(d_in) + 1e-9) + d_out / (np.linalg.norm(d_out) + 1e-9)
        v[i] = speed * d / (np.linalg.norm(d) + 1e-9)
    return v


def optimize_velocities(
    p: NDArray,
    v_start: NDArray,
    dirs: NDArray,
    f_max: float,
    v_init: NDArray | None = None,
    iters: int = OPT_ITERS,
    vmax: NDArray | None = None,
) -> tuple[NDArray, Segments]:
    """Minimise the total time over the waypoint velocities (projected gradient descent).

    Args:
        p: Waypoints (n+1, 3). p[0] is the current state.
        v_start: Fixed velocity at p[0].
        dirs: Unit gate normals (n+1, 3) oriented along the travel direction; NaN rows mark
            waypoints with free 3D velocity.
        f_max: Mass-normalised thrust limit.
        v_init: Optional warm start of the waypoint velocities (n+1, 3).
        iters: Maximum number of gradient iterations.
        vmax: Optional speed bound per waypoint (n+1,). Defaults to V_MAX.

    Returns:
        Optimal waypoint velocities and the corresponding segments.
    """
    vmax = np.full(len(p), V_MAX) if vmax is None else np.asarray(vmax, dtype=float)
    fixed = np.zeros(len(p), dtype=bool)
    fixed[0] = True
    v = _initial_velocities(p, dirs) if v_init is None else v_init.copy()
    v[0] = v_start
    v = _project(v, dirs, fixed, vmax)
    bounds = initial_bounds(len(p) - 1, f_max)
    n_inner = max(iters // N_OUTER, 1)
    for _ in range(N_OUTER):
        # (1) thrust decomposition at the current velocities, (2) velocity descent with the
        # per-segment bounds held fixed (consistent objective and gradient)
        bounds = refine_bounds(p, v, bounds, f_max, N_DECOMP)
        seg = solve_segments(p, v, bounds)
        cost = seg.dt.sum()
        step = OPT_STEP0
        for _ in range(n_inner):
            grad = np.zeros_like(v)
            grad[:-1] += seg.grad0
            grad[1:] += seg.grad1
            grad[fixed] = 0.0
            v_new = _project(v - step * grad, dirs, fixed, vmax)
            if np.max(np.abs(v_new - v)) < 1e-4:
                break
            seg_new = solve_segments(p, v_new, bounds)
            cost_new = seg_new.dt.sum()
            if cost_new < cost - 1e-7:
                v, seg, cost = v_new, seg_new, cost_new
                step *= 1.5
            else:
                step *= 0.4
                if step < 1e-3:
                    break
    return v, seg


# ---------------------------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------------------------
class PMMTrajectory:
    """Piecewise bang-bang trajectory through waypoints, evaluated in closed form."""

    def __init__(self, p: NDArray, v: NDArray, seg: Segments, t_offset: float = 0.0):
        """Store the waypoints, their velocities and the synchronised segments.

        Args:
            p: Waypoints (n+1, 3).
            v: Waypoint velocities (n+1, 3).
            seg: Segments between the waypoints.
            t_offset: Absolute time of p[0].
        """
        self.p, self.v, self.seg = p, v, seg
        self.t_knots = t_offset + np.concatenate([[0.0], np.cumsum(seg.dt)])
        self.t_start, self.t_end = float(self.t_knots[0]), float(self.t_knots[-1])

    @property
    def duration(self) -> float:
        """Total duration of the trajectory."""
        return self.t_end - self.t_start

    def segment_index(self, t: float | NDArray) -> NDArray:
        """Index of the segment active at time t."""
        i = np.searchsorted(self.t_knots, t, side="right") - 1
        return np.clip(i, 0, len(self.seg.dt) - 1)

    def __call__(self, t: float | NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Position, velocity and acceleration at (absolute) time(s) t."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        i = self.segment_index(t)
        s = self.seg
        tau = np.clip(t - self.t_knots[i], 0.0, s.dt[i])[:, None]
        a, t1, p0, v0 = s.acc[i], s.t1[i], s.p0[i], s.v0[i]
        first = tau <= t1
        ps = p0 + v0 * t1 + 0.5 * a * t1**2
        vs = v0 + a * t1
        tb = tau - t1
        pos = np.where(first, p0 + v0 * tau + 0.5 * a * tau**2, ps + vs * tb - 0.5 * a * tb**2)
        vel = np.where(first, v0 + a * tau, vs - a * tb)
        acc = np.where(first, a, -a)
        # Beyond the end: continue with the final velocity
        after = (t > self.t_end)[:, None]
        pos = np.where(after, self.p[-1] + self.v[-1] * (t - self.t_end)[:, None], pos)
        vel = np.where(after, self.v[-1], vel)
        acc = np.where(after, 0.0, acc)
        return pos, vel, acc


class _Chain:
    """Two trajectories glued at a switch time (used while a re-plan is being committed)."""

    def __init__(self, first, second, t_switch: float):  # noqa: ANN001
        self.first, self.second, self.t_switch = first, second, t_switch

    def __call__(self, t: NDArray | float) -> tuple[NDArray, NDArray, NDArray]:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        a, b = self.first(t), self.second(t)
        m = (t < self.t_switch)[:, None]
        return tuple(np.where(m, x, y) for x, y in zip(a, b))


def plan_pmm(
    p: NDArray,
    v_start: NDArray,
    dirs: NDArray,
    f_max: float = F_MAX,
    v_init: NDArray | None = None,
    iters: int = OPT_ITERS,
    t_offset: float = 0.0,
    vmax: NDArray | None = None,
) -> PMMTrajectory:
    """Plan a minimum-time PMM trajectory through the waypoints p starting with v_start."""
    v, seg = optimize_velocities(p, v_start, dirs, f_max, v_init, iters, vmax)
    return PMMTrajectory(p, v, seg, t_offset)


# ---------------------------------------------------------------------------------------------
# Collision checking against the track geometry (used for diagnostics and path repair)
# ---------------------------------------------------------------------------------------------
DRONE_HALF = 0.07  # [m] horizontal half size of the drone collision box (0.14 x 0.14 m)
DRONE_HALF_Z = 0.04  # [m] vertical half size of the box (0.02) plus an allowance for tilt
OBST_R = 0.015  # [m] radius of the obstacle poles
GATE_OUTER = 0.36  # [m] half size of the gate frame (outer)
GATE_INNER = 0.20  # [m] half size of the gate opening
GATE_DEPTH = 0.01  # [m] half thickness of the gate frame


def _box_extent(angle: NDArray) -> NDArray:
    """Half extent of the square drone box along a direction at ``angle`` to its x axis."""
    return DRONE_HALF * (np.abs(np.cos(angle)) + np.abs(np.sin(angle)))


def collision_distances(
    pos: NDArray,
    gates_pos: NDArray,
    gates_quat: NDArray,
    obstacles_pos: NDArray,
    yaw: NDArray | float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Clearance of the drone collision box to obstacle poles and gate frames.

    The race environment uses a 0.14 x 0.14 x 0.04 m box as the drone's collision geometry, so the
    clearance depends on the yaw of the drone relative to the object.

    Args:
        pos: Sampled drone positions (m, 3).
        gates_pos: Gate positions (n_g, 3).
        gates_quat: Gate orientations (n_g, 4), xyzw.
        obstacles_pos: Obstacle top positions (n_o, 3).
        yaw: Drone yaw per sample (m,) or scalar.

    Returns:
        Clearances (m, n_o) to the poles and (m, n_g) to the gate frames. Negative = collision.
    """
    yaw = np.broadcast_to(np.asarray(yaw, dtype=float), (len(pos),))
    rel = pos[:, None, :2] - obstacles_pos[None, :, :2]
    d_xy = np.linalg.norm(rel, axis=-1)
    ext = _box_extent(np.arctan2(rel[..., 1], rel[..., 0]) - yaw[:, None])
    d_obs = d_xy - OBST_R - ext
    above = pos[:, None, 2] - DRONE_HALF_Z - obstacles_pos[None, :, 2]
    d_obs = np.where(above > 0, np.maximum(d_obs, above), d_obs)
    d_gate = np.empty((len(pos), len(gates_pos)))
    for j, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
        rot = R.from_quat(gq)
        loc = rot.apply(pos - gp, inverse=True)
        gate_yaw = rot.as_euler("xyz")[2]
        e_xy = _box_extent(yaw - gate_yaw)  # same along the gate normal and in-plane
        x, y, z = np.abs(loc[:, 0]), np.abs(loc[:, 1]), loc[:, 2]
        za = np.abs(z)
        gx = x - GATE_DEPTH - e_xy  # > 0: the box does not reach the frame plane
        inner = np.minimum(GATE_INNER - y - e_xy, GATE_INNER - za - DRONE_HALF_Z)
        outer = np.maximum(y - e_xy - GATE_OUTER, za - DRONE_HALF_Z - GATE_OUTER)
        gap = np.maximum(inner, outer)  # in-plane clearance to the frame bars
        d = np.where(gx > 0, np.where(gap > 0, np.hypot(gx, gap), gx), gap)
        # Gate stand below the frame (0.1 x 0.1 m column)
        sx, sy = np.maximum(x - 0.05 - e_xy, 0.0), np.maximum(y - 0.05 - e_xy, 0.0)
        sz = np.maximum(z - DRONE_HALF_Z + GATE_OUTER, 0.0)
        d_stand = np.sqrt(sx**2 + sy**2 + sz**2)
        d_gate[:, j] = np.minimum(d, d_stand)
    return d_obs, d_gate


# ---------------------------------------------------------------------------------------------
# Controller parameters
# ---------------------------------------------------------------------------------------------
# Unseen gates: the true pose is revealed only within the sensor range (0.7 m). Until then the
# plan approaches the gate straight along its nominal normal and slowly enough to correct a pose
# error of up to ~0.15 m after the reveal.
V_GATE_UNSEEN = 1.8  # [m/s] max. speed through gates whose true pose has not been observed yet
APPROACH_D = (0.3, 0.4, 0.4, 0.0, 0.0)  # [m] approach point before each unseen gate visit
V_APPROACH = 1.8  # [m/s] max. speed at the approach points
D_SLOW = 0.8  # [m] helper waypoints this close to the next unseen gate get the V_GATE_UNSEEN cap
GATE_Z_OFFSET = 0.0  # [m] height offset of the gate waypoints
TAKEOFF_HEIGHT = 0.25  # [m] first waypoint above the start position
# Tracking (full-state command to the onboard Mellinger controller)
FF_LOOKAHEAD = 0.12  # [s] the acceleration feed-forward is taken this far ahead (attitude lag)
ACC_SMOOTH_W = 0.12  # [s] the feed-forward is the mean acceleration over this window (0 = off)
KP_EXTRA = 5.0  # [1/s^2] extra position feedback through the acceleration channel
KD_EXTRA = 1.0  # [1/s] extra velocity feedback through the acceleration channel
ALIGN_YAW = False  # yaw the (square) drone box parallel to the next gate's frame
YAW_RATE = 3.0  # [rad/s] rate limit of the yaw command
THRUST_COMP = True  # invert the thrust scaling of the onboard Mellinger controller
SCALE_ADAPT = 0.5  # [1/(m s)] adaptation rate of the thrust scale estimate (0 = off)
# Progress adaptation: the reference time slows down when the drone lags behind
LAG_FREE = 0.08  # [m] along-track lag tolerated without slowing the reference
LAG_SCALE = 0.25  # [m] additional lag at which the rate reaches RATE_MIN
RATE_MIN = 0.4
# Re-planning
REPLAN = True  # re-plan when observed gate/obstacle poses change
T_COMMIT = 0.08  # [s] a new plan starts from the old reference this far ahead (attitude lag)
REPLAN_TOL = 0.01  # [m] waypoint shift below which a still collision-free plan is kept
POSE_TOL = 1e-3  # change of an observed pose that triggers a re-plan
D_SKIP = 0.3  # [m] helper waypoints closer than this may be skipped when re-planning
T_SKIP = 0.2  # [s] ... or reached within this time
CLEAR_MIN = 0.04  # [m] required clearance of a candidate plan to known obstacles and gates
CLEAR_DT = 0.02  # [s] sampling of the clearance check
R_AVOID_MIN = 0.22  # [m] min. horizontal distance of an avoidance point to its observed pole
R_DETOUR = 0.25  # [m] distance of an automatically inserted detour point to the pole axis
GATE_DETOUR = 0.58  # [m] in-plane half size of the square a gate detour point is moved onto
D_GATE_AXIS = 0.25  # [m] distance of a detour point on the axis of the gate being crossed
N_REPAIR = 3  # max. detour insertions per (re-)plan
T_IGNORE = 0.1  # [s] the clearance of the first part of a new plan cannot be changed: ignore it

# Track specific helper points for level 2 (adapted from pathplanner_cs / state_pp_racer).
AV2_Z = 1.0  # [m] height of the U-turn point behind gate 3
SUPPORT_POINTS = np.array([[0.5, 0.75, 1.2], [-0.5, -0.05, 0.7], [0.5, 1.25, 1.2]])
# Obstacle avoidance points: (obstacle index, dx, dy, absolute z), anchored to the *nominal* pole
# positions and pushed away from the observed pole only if it is closer than R_AVOID_MIN. This
# keeps re-planning shifts small when a pole is revealed.
AVOID = [
    (0, 0.0, -0.3, 0.7),  # south of obstacle 1, before gate 1
    (1, 0.25, -0.3, 1.0),  # east of obstacle 2, before gate 2
    (2, 0.1, 0.35, None),  # north of obstacle 3, U-turn behind gate 3 (z = AV2_Z)
    (3, 0.05, 0.22, 1.2),  # north of obstacle 4, before gate 4
    (1, 0.25, -0.3, 1.2),  # east of obstacle 2, before gate 2 (second visit)
]


class PMMRacer(Controller):
    """Minimum-time point-mass-model trajectory with full-state feed-forward and re-planning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialise the controller and plan the initial trajectory.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        # Thrust model of the onboard Mellinger controller vs. the simulated drone. The controller
        # uses a mass of 0.0393 kg and the PWM constant ``mass_thrust`` which amplifies the force
        # by ~1.61, so a commanded acceleration is realised ~1.46x too strong. Pre-scale it.
        dyn = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        mel = load_fn_params(state2attitude, config.sim.drone)
        pwm_gain = float(mel["mass_thrust"]) * 4 * mel["thrust_max"] / mel["pwm_max"]
        self._scale0 = float(dyn["mass"]) / (pwm_gain * float(mel["mass"])) if THRUST_COMP else 1.0
        self._scale = self._scale0

        self._start_pos = np.asarray(obs["pos"], dtype=float).copy()
        self._obst_nominal = np.asarray(obs["obstacles_pos"], dtype=float).copy()
        self._read_track(obs)
        self._t = 0.0  # reference time
        self._rate = 1.0  # progress rate of the reference time
        self._tick = 0
        self._finished = False
        self._yaw_cmd = 0.0
        self._prev, self._t_switch = None, -np.inf
        self._wp_vel: dict[str, NDArray] = {}  # last optimised velocity per waypoint key
        self._detours: list[dict] = []  # automatically inserted detour points around poles
        self.debug: list[dict] = []  # one entry per executed re-plan (for analysis)
        keys = self._waypoints()[0]
        self._plan(self._start_pos, np.zeros(3), [keys[1]], OPT_ITERS, (1.0, 1.5, 2.0, 2.5))

    # -----------------------------------------------------------------------------------------
    # Track and waypoints
    # -----------------------------------------------------------------------------------------
    def _read_track(self, obs: dict[str, NDArray[np.floating]]):
        self._gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()
        self._gates_seen = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._gate_seq = np.asarray(obs["gate_sequence"], dtype=int).copy()
        self._gate_dir = np.asarray(obs["gate_sequence_direction"], dtype=float).copy()

    def _track_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        return bool(
            np.max(np.abs(obs["gates_pos"] - self._gates_pos)) > POSE_TOL
            or np.max(np.abs(obs["gates_quat"] - self._gates_quat)) > POSE_TOL
            or np.max(np.abs(obs["obstacles_pos"] - self._obstacles_pos)) > POSE_TOL
            or np.any(np.asarray(obs["gates_visited"], dtype=bool) != self._gates_seen)
        )

    def _gate_normal(self, gate: int, direction: float) -> NDArray:
        """Gate normal (local x axis) oriented along the passing direction."""
        return direction * R.from_quat(self._gates_quat[gate]).apply([1.0, 0.0, 0.0])

    def _gate_yaw(self, gate: int) -> float:
        return float(R.from_quat(self._gates_quat[gate]).as_euler("xyz")[2])

    def _key_yaws(self, keys: list[str]) -> NDArray:
        """Planned drone yaw on the way to each waypoint: that of the next gate (box alignment)."""
        if not ALIGN_YAW:
            return np.zeros(len(keys))
        yaws = np.zeros(len(keys))
        gate_keys = [k for k in keys if k[0] == "g" and k[1:].isdigit()]
        current = self._gate_yaw(self._gate_seq[int(gate_keys[-1][1:])]) if gate_keys else 0.0
        for i in range(len(keys) - 1, -1, -1):
            k = keys[i]
            if k[0] == "g" and k[1:].isdigit():
                current = self._gate_yaw(self._gate_seq[int(k[1:])])
            yaws[i] = current
        return yaws

    def _avoid_points(self) -> NDArray:
        pts = []
        for i, dx, dy, z in AVOID:
            p = np.array([self._obst_nominal[i, 0] + dx, self._obst_nominal[i, 1] + dy,
                          AV2_Z if z is None else z])
            d = p[:2] - self._obstacles_pos[i, :2]
            r = np.linalg.norm(d)
            if r < R_AVOID_MIN:
                p[:2] = self._obstacles_pos[i, :2] + d / max(r, 1e-6) * R_AVOID_MIN
            pts.append(p)
        return np.array(pts)

    def _waypoints(self) -> tuple[list[str], NDArray, NDArray, NDArray, NDArray]:
        """Waypoint list of the full lap with stable keys.

        Returns:
            Keys, points (m, 3), velocity cone axes (m, 3) (NaN = free velocity), speed bounds (m,)
            and a gate flag (m,).
        """
        av = self._avoid_points()
        sp = SUPPORT_POINTS
        # The support point before gate 3 is only needed once its approach point is gone
        g2_unseen = not self._gates_seen[self._gate_seq[2]]
        sp1 = [] if (g2_unseen and APPROACH_D[2] > 0) else [("sp1", sp[1])]
        seq = [
            ("start", self._start_pos), ("takeoff", self._start_pos + [0, 0, TAKEOFF_HEIGHT]),
            ("av0", av[0]), ("g0", 0), ("av1", av[1]), ("g1", 1), ("sp0", sp[0]), *sp1,
            ("g2", 2), ("av2", av[2]), ("av3", av[3]), ("g3", 3), ("av4", av[4]), ("g4", 4),
            ("sp2", sp[2]),
        ]
        keys, pts, dirs, vmax, is_gate = [], [], [], [], []

        def add(key: str, p: NDArray, d: NDArray | None, vm: float, gate: bool):
            keys.append(key)
            pts.append(p)
            dirs.append(np.full(3, np.nan) if d is None else d)
            vmax.append(vm)
            is_gate.append(gate)

        for key, item in seq:
            if isinstance(item, int):  # gate visit
                gate = self._gate_seq[item]
                n = self._gate_normal(gate, self._gate_dir[item])
                gp = self._gates_pos[gate] + [0.0, 0.0, GATE_Z_OFFSET]
                seen = self._gates_seen[gate]
                d_app = APPROACH_D[item] if item < len(APPROACH_D) else 0.0
                if not seen and d_app > 0:
                    add(f"a{item}", gp - d_app * n, n, V_APPROACH, False)
                add(key, gp, n, V_MAX if seen else V_GATE_UNSEEN, True)
            else:
                add(key, np.asarray(item, dtype=float), None, V_MAX, False)
        # Slow zone: helper points shortly before an unseen gate are capped like the gate itself
        last_gate = -1
        for i, (k, g) in enumerate(zip(keys, is_gate)):
            if not g:
                continue
            visit = int(k[1:])
            if not self._gates_seen[self._gate_seq[visit]]:
                for j in range(last_gate + 1, i):
                    if np.linalg.norm(pts[j] - pts[i]) < D_SLOW:
                        vmax[j] = min(vmax[j], V_GATE_UNSEEN)
            last_gate = i
        for det in self._detours:  # in creation order, each before its successor waypoint
            if det["before"] not in keys:
                continue
            j = keys.index(det["before"])
            p = det["pos"].copy()
            if det["obst"] >= 0:
                d = p[:2] - self._obstacles_pos[det["obst"], :2]
                r = np.linalg.norm(d)
                if r < R_DETOUR:  # the observed pole is closer than planned: push the point away
                    p[:2] = self._obstacles_pos[det["obst"], :2] + d / max(r, 1e-6) * R_DETOUR
            keys.insert(j, det["key"])
            pts.insert(j, p)
            dirs.insert(j, det["dir"])
            vmax.insert(j, V_MAX)
            is_gate.insert(j, False)
        return keys, np.array(pts), np.array(dirs), np.array(vmax), np.array(is_gate)

    # -----------------------------------------------------------------------------------------
    # Planning
    # -----------------------------------------------------------------------------------------
    def _plan(self, p_start: NDArray, v_start: NDArray, first_keys: list[str], iters: int,
              cold_speeds: tuple[float, ...] = ()):
        """Plan from (p_start, v_start) through the remaining waypoints and activate the result."""
        traj, score, traj_keys = self._search(p_start, v_start, first_keys, iters, cold_speeds,
                                              N_REPAIR)
        for j, k in enumerate(traj_keys):
            self._wp_vel[k] = traj.v[j + 1].copy()
        self._traj = traj
        self._traj_keys = traj_keys
        keys, self._wps, dirs, _, _ = self._waypoints()
        self._traj_dirs = dirs[keys.index(traj_keys[0]):]
        self._clear = score

    def _search(self, p_start: NDArray, v_start: NDArray, first_keys: list[str], iters: int,
                cold_speeds: tuple[float, ...], repairs: int,
                ) -> tuple[PMMTrajectory, tuple[bool, float], list[str]]:
        """Best candidate plan, with collision repair by detour points.

        Candidates differ in the first waypoint (``first_keys``: skip nearly reached helper points)
        and in the initial guess of the waypoint velocities (warm start from the previous plan,
        warm start with the first waypoint velocity set to the current velocity, cold starts). The
        fastest candidate that keeps CLEAR_MIN clearance to all known obstacles and gates wins. If
        none does, a detour point is inserted and the search is repeated; the detour is kept only
        if it improves the result.

        Returns:
            Trajectory, score (collision flag, time or negative clearance) and waypoint keys.
        """
        keys, wps, dirs, vmax, _ = self._waypoints()
        best, best_score, best_base = None, None, None
        for first in first_keys:
            base = keys.index(first)
            p = np.vstack([p_start, wps[base:]])
            d = np.vstack([np.full(3, np.nan), dirs[base:]])
            vm = np.concatenate([[V_MAX], vmax[base:]])
            inits: list[NDArray] = []
            if not cold_speeds and any(k in self._wp_vel for k in keys[base:]):
                heur = _initial_velocities(p, d)
                rest = [self._wp_vel.get(k, heur[j + 1]) for j, k in enumerate(keys[base:])]
                v_warm = np.vstack([v_start] + rest)
                inits.append(v_warm)
                v_alt = v_warm.copy()
                v_alt[1] = v_start
                inits.append(v_alt)
            for speed in cold_speeds or ((V_INIT,) if not inits else ()):
                inits.append(_initial_velocities(p, d, speed))
            for v_init in inits:
                traj = plan_pmm(p, v_start, d, F_MAX, v_init, iters, t_offset=self._t, vmax=vm)
                clear = self._clearance(traj, keys[base:])
                score = (clear < CLEAR_MIN, traj.t_end if clear >= CLEAR_MIN else -clear)
                if best is None or score < best_score:
                    best, best_score, best_base = traj, score, base
        best_keys = keys[best_base:]
        n_det = len(self._detours)
        if best_score[0] and repairs > 0 and self._add_detour(best, best_keys):
            det = self._detours[-1]
            first = [det["key"] if k == det["before"] else k for k in first_keys]
            cand = self._search(p_start, v_start, first, iters, cold_speeds, repairs - 1)
            if cand[1] < best_score:
                return cand
            del self._detours[n_det:]  # the repair did not help
        return best, best_score, best_keys

    def _samples(
        self, traj: PMMTrajectory, traj_keys: list[str]
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Positions, velocities, times, segment indices and planned yaw of the clearance check."""
        ts = np.arange(traj.t_start + T_IGNORE, traj.t_end, CLEAR_DT)
        pos, vel, _ = traj(ts)
        seg = traj.segment_index(ts)
        return pos, vel, ts, seg, self._key_yaws(traj_keys)[seg]

    def _clearance(self, traj: PMMTrajectory, traj_keys: list[str]) -> float:
        """Minimum clearance of the planned trajectory to the known obstacles and gate frames."""
        pos, _, ts, _, yaw = self._samples(traj, traj_keys)
        if len(ts) == 0:
            return np.inf
        d_obs, d_gate = collision_distances(pos, self._gates_pos, self._gates_quat,
                                            self._obstacles_pos, yaw)
        return float(min(d_obs.min(), d_gate.min()))

    def _add_detour(self, traj: PMMTrajectory, traj_keys: list[str]) -> bool:
        """Insert a detour point next to the pole or gate frame the trajectory comes closest to.

        Poles: the point is placed R_DETOUR from the pole axis on the side the path passes.
        Gate frames passed on the outside: the point is moved in the gate plane onto a square of
        half size GATE_DETOUR (above the gate or beside it, never below because of the stand).
        Collisions inside the gate opening (crossing errors) cannot be fixed by a detour.

        Returns:
            True if a detour was added, False otherwise.
        """
        pos, vel, ts, _, yaw = self._samples(traj, traj_keys)
        if len(ts) == 0:
            return False
        d_obs, d_gate = collision_distances(pos, self._gates_pos, self._gates_quat,
                                            self._obstacles_pos, yaw)
        if min(d_obs.min(), d_gate.min()) >= CLEAR_MIN:
            return False
        if d_obs.min() <= d_gate.min():
            k, i = np.unravel_index(np.argmin(d_obs), d_obs.shape)
            o = self._obstacles_pos[i]
            u = pos[k, :2] - o[:2]
            if np.linalg.norm(u) < 0.02:  # path through the pole centre: go to the left
                u = np.array([-vel[k, 1], vel[k, 0]])
            u /= np.linalg.norm(u) + 1e-9
            p = np.array([o[0] + R_DETOUR * u[0], o[1] + R_DETOUR * u[1], pos[k, 2]])
            obst, cone = int(i), np.full(3, np.nan)
            seg = int(np.atleast_1d(traj.segment_index(ts[k]))[0])
        else:
            k, j = np.unravel_index(np.argmin(d_gate), d_gate.shape)
            seg = int(np.atleast_1d(traj.segment_index(ts[k]))[0])
            rot = R.from_quat(self._gates_quat[j])
            loc = rot.apply(pos[k] - self._gates_pos[j], inverse=True)
            if max(abs(loc[1]), abs(loc[2])) < GATE_INNER:  # crossing error, not a detour case
                return False
            obst, cone = -1, np.full(3, np.nan)
            nxt = traj_keys[seg]
            prv = traj_keys[seg - 1] if seg > 0 else None
            visit_n = int(nxt[1:]) if nxt[0] == "g" and nxt[1:].isdigit() else -1
            visit_p = int(prv[1:]) if prv and prv[0] == "g" and prv[1:].isdigit() else -1
            if visit_n >= 0 and self._gate_seq[visit_n] == j:  # clipped while approaching it
                n = self._gate_normal(j, self._gate_dir[visit_n])
                p, cone = self._gates_pos[j] - D_GATE_AXIS * n, n
            elif visit_p >= 0 and self._gate_seq[visit_p] == j:  # clipped while leaving it
                n = self._gate_normal(j, self._gate_dir[visit_p])
                p, cone = self._gates_pos[j] + D_GATE_AXIS * n, n
            else:  # passing by: move the point in the gate plane above or beside the frame
                if loc[2] > -0.5 * abs(loc[1]):  # never below (stand)
                    loc[2] = GATE_DETOUR if loc[2] >= abs(loc[1]) else loc[2]
                if abs(loc[2]) < GATE_DETOUR:
                    side = np.sign(loc[1]) if abs(loc[1]) > 1e-3 else 1.0
                    loc[1] = side * max(abs(loc[1]), GATE_DETOUR)
                p = self._gates_pos[j] + rot.apply(loc)
        key = f"det{len(self._detours)}"
        self._detours.append(dict(key=key, before=traj_keys[seg], obst=obst, pos=p, dir=cone))
        self._wp_vel[key] = vel[k].copy()
        return True

    def _ref(self, t: NDArray | float) -> tuple[NDArray, NDArray, NDArray]:
        """Reference at time(s) t: the previous plan before the switch time, the new one after."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self._prev is None or t.min() >= self._t_switch:
            return self._traj(t)
        old, new = self._prev(t), self._traj(t)
        use_old = (t < self._t_switch)[:, None]
        return tuple(np.where(use_old, o, n) for o, n in zip(old, new))

    def _plan_needed(self) -> bool:
        """A re-plan is needed if the remaining waypoints changed or the plan became unsafe."""
        keys, wps, dirs, _, _ = self._waypoints()
        if self._traj_keys[-1] not in keys or self._traj_keys[0] not in keys:
            return True
        remaining = keys[keys.index(self._traj_keys[0]):]
        if remaining != self._traj_keys:
            return True
        i0 = keys.index(self._traj_keys[0])
        moved = np.linalg.norm(wps[i0:] - self._traj.p[1:], axis=1).max() > REPLAN_TOL
        turned = np.nan_to_num(np.abs(dirs[i0:] - self._traj_dirs)).max() > 0.02
        return bool(moved or turned or self._clearance(self._traj, self._traj_keys) < CLEAR_MIN)

    def _replan(self):
        """Re-plan from the (committed) reference state through the remaining waypoints."""
        if not self._plan_needed():
            return
        t_r = self._t + T_COMMIT
        old = self._traj
        pos, vel, _ = (x[0] for x in self._ref(t_r))
        i = int(np.atleast_1d(self._traj.segment_index(t_r))[0])
        keys, wps, _, _, is_gate = self._waypoints()
        # The next waypoint may have disappeared (approach point of a gate that was just seen)
        while i < len(self._traj_keys) and self._traj_keys[i] not in keys:
            i += 1
        if i >= len(self._traj_keys):
            return
        base = keys.index(self._traj_keys[i])
        # Helper waypoints (not gates) that are nearly reached may be skipped: forcing the drone
        # through a shifted point it is about to reach causes loops.
        skip = base
        while skip < len(keys) - 1 and not is_gate[skip]:
            k = keys[skip]
            t_reach = np.inf
            if k in self._traj_keys:
                t_reach = self._traj.t_knots[self._traj_keys.index(k) + 1] - t_r
            behind = (wps[skip] - pos) @ vel < 0.0  # already passed (e.g. after a pole reveal)
            if np.linalg.norm(wps[skip] - pos) >= D_SKIP and t_reach >= T_SKIP and not behind:
                break
            skip += 1
        first = [keys[base]] if skip == base else [keys[base], keys[skip]]
        if skip != base and (wps[base] - pos) @ vel < 0.0:  # never turn back to a passed point
            first = [keys[skip]]
        old_end = self._traj.t_end
        t_now = self._t
        self._t = t_r  # the new plan starts at t_r
        self._plan(pos, vel, first, OPT_ITERS_WARM)
        self._t = t_now
        if self._prev is not None and t_now < self._t_switch:  # previous switch still pending
            self._prev = _Chain(self._prev, old, self._t_switch)
        else:
            self._prev = old
        self._t_switch = t_r
        self.debug.append(dict(t=self._t, first=first, chosen=self._traj_keys[0], old_end=old_end,
                               new_end=self._traj.t_end, clear=self._clear))

    # -----------------------------------------------------------------------------------------
    # Controller interface
    # -----------------------------------------------------------------------------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the full state command [pos, vel, acc, quat, body rates].

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz].
        """
        if int(obs["n_gates_passed"]) >= len(obs["gate_sequence"]):
            self._finished = True
        if REPLAN and self._track_changed(obs) and not self._finished:
            self._read_track(obs)
            self._replan()
        pos, vel, _ = self._ref(self._t)
        t_ff = self._t + FF_LOOKAHEAD * self._rate
        if ACC_SMOOTH_W > 0:  # mean acceleration over a window = velocity difference
            half = 0.5 * ACC_SMOOTH_W
            v_lo, v_hi = self._ref(np.array([t_ff - half, t_ff + half]))[1]
            acc = (v_hi - v_lo) / ACC_SMOOTH_W
        else:
            acc = self._ref(t_ff)[2][0]
        pos, vel, acc = pos[0], vel[0] * self._rate, acc * self._rate**2
        # Extra feedback through the acceleration channel (the onboard gains are fixed and low)
        acc = acc + KP_EXTRA * (pos - obs["pos"]) + KD_EXTRA * (vel - obs["vel"])
        # Invert the thrust scaling of the onboard controller
        g = np.array([0.0, 0.0, G])
        acc_cmd = self._scale * (acc + g) - g
        # Yaw: align the square collision box with the frame of the next gate (mod 90 deg)
        visit = int(obs["n_gates_passed"])
        if ALIGN_YAW and visit < len(self._gate_seq):
            target = self._gate_yaw(self._gate_seq[visit])
            delta = (target - self._yaw_cmd + np.pi / 4) % (np.pi / 2) - np.pi / 4
            self._yaw_cmd += float(np.clip(delta, -YAW_RATE / self._freq, YAW_RATE / self._freq))
        quat = R.from_euler("z", self._yaw_cmd).as_quat()
        return np.concatenate((pos, vel, acc_cmd, quat, np.zeros(3)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the reference time and adapt the thrust scale.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        p_ref, v_ref, _ = (x[0] for x in self._ref(self._t))
        if SCALE_ADAPT > 0:  # steady height error -> unknown mass (randomised in level 2)
            e_z = float(p_ref[2] - obs["pos"][2])
            self._scale = float(np.clip(self._scale + SCALE_ADAPT * e_z / self._freq,
                                        0.8 * self._scale0, 1.25 * self._scale0))
        speed = float(np.linalg.norm(v_ref))
        lag = float((p_ref - obs["pos"]) @ v_ref / speed) if speed > 0.1 else 0.0
        self._rate = float(np.clip(1.0 - (lag - LAG_FREE) / LAG_SCALE, RATE_MIN, 1.0))
        self._t += self._rate / self._freq
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0
        self._t = 0.0
        self._rate = 1.0
        self._yaw_cmd = 0.0
        self._prev, self._t_switch = None, -np.inf
        self._scale = self._scale0
        self._finished = False

    def render_callback(self, sim: Sim):
        """Visualise the planned trajectory, the waypoints and the current setpoint."""
        draw_points(sim, self._ref(self._t)[0], rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        draw_points(sim, self._wps, rgba=(0.0, 0.0, 1.0, 1.0), size=0.015)
        ts = np.linspace(self._traj.t_start, self._traj.t_end, 150)
        draw_line(sim, self._traj(ts)[0], rgba=(0.0, 1.0, 0.0, 1.0))
