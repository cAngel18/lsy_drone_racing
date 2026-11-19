"""Fast racing controller on the full-state (Mellinger) interface, built on ``pathplanner_cs``.

``pathplanner_cs`` sends only a position setpoint from a cubic spline with uniformly spaced knot
times (12 s total). The onboard Mellinger controller then has to generate all acceleration from
position error, which lags by tenths of a second and limits how fast the track can be flown.
This controller keeps its proven waypoint layout but uses the rest of the state command:

1. **Path, not trajectory.** The waypoints (gates + obstacle avoidance points + support points)
   are joined by a chord-length parameterised Hermite spline. At every gate centre the tangent is
   pinned to the gate normal so gates are crossed straight.
2. **Dynamics-aware timing.** A time-optimal speed profile v(s) along the arc length respects a
   speed limit, a lateral/total acceleration limit and the thrust limit |a + g| <= F_MAX of the
   drone (forward/backward pass). This replaces the uniform knot times.
3. **Full state reference.** Position, velocity *and* acceleration of the reference are sent, so
   the Mellinger controller tilts the drone ahead of time (differential flatness: acceleration ->
   thrust vector -> attitude) instead of reacting to position errors.
4. **Thrust model inversion.** The simulated Mellinger controller uses a mass of 0.0393 kg and
   maps thrust through the firmware PWM constant ``mass_thrust``, which amplifies every force by
   ~1.61. With the true mass (0.0434 kg) the commanded acceleration is realised ~1.46x too strong
   and the drone hovers ~0.1 m above its setpoint (the reason for the ``-0.1`` gate height hack in
   ``pathplanner_cs``). The acceleration command is pre-scaled so that the realised thrust matches
   m * (a_ref + g) exactly, with an online estimate of the scale for the randomised mass.
5. **Progress variable.** The reference is indexed by a progress s that advances with v(s) and
   slows down when the drone lags behind, instead of running on wall-clock time.

Re-planning: whenever an observed gate or obstacle pose changes (sensor range 0.7 m) the full path
is rebuilt and the progress is re-projected onto the new path, which keeps the reference continuous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.control.core import load_fn_params
from crazyflow.control.mellinger import state2attitude
from crazyflow.dynamics import load_params as load_dynamics_params
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------------------------
# Tunable parameters (module level so that a test harness can override them)
# ---------------------------------------------------------------------------------------------
G = 9.81  # [m/s^2]
# Speed profile limits of the reference
V_MAX = 2.5  # [m/s]
A_MAX = 7.0  # [m/s^2] total reference acceleration (tangential + centripetal)
A_LAT_MAX = 6.0  # [m/s^2] centripetal acceleration
F_MAX = 13.0  # [m/s^2] mass-normalised thrust |a + g| of the reference (hardware max ~18.4)
V_START = 0.2  # [m/s] minimum progress speed (v(0) = 0)
V_UNSEEN = 1.5  # [m/s] speed cap near gates whose true pose has not been observed yet
UNSEEN_R = 0.5  # [m] radius around such a gate in which the cap applies
# Path geometry
GATE_Z_OFFSET = 0.0  # [m] height offset of the gate waypoints (pathplanner_cs used -0.1)
DS = 0.02  # [m] arc length resolution of the sampled path
TAKEOFF_HEIGHT = 0.2  # [m]
# Tracking
FF_LOOKAHEAD = 0.1  # [s] the acceleration feed-forward is taken this far ahead on the path
POS_LOOKAHEAD = 0.0  # [s] the position/velocity reference is taken this far ahead (latency)
THRUST_COMP = True  # invert the Mellinger thrust scaling (see module docstring)
SCALE_ADAPT = 0.5  # [1/(m s)] adaptation rate of the thrust scale estimate (0 = off)
KP_EXTRA = np.array([0.0, 0.0, 0.0])  # [1/s^2] extra position feedback through the acc channel
KD_EXTRA = np.array([0.0, 0.0, 0.0])  # [1/s] extra velocity feedback through the acc channel
# Progress adaptation
LAG_FREE = 0.05  # [m] along-track lag tolerated without slowing the reference
LAG_SCALE = 0.25  # [m] additional lag at which the rate reaches RATE_MIN
RATE_MIN = 0.3
POSE_TOL = 1e-3  # change of an observed pose that triggers a re-plan

# Support points of pathplanner_cs (absolute positions).
SUPPORT_POINTS = np.array([[0.5, 0.75, 1.2], [-0.5, -0.05, 0.7], [0.5, 1.25, 1.2]])


# Obstacle avoidance points of pathplanner_cs: (obstacle index, dx, dy, absolute z). Order matters,
# see ``StatePPRacer._waypoints``.
AVOID = [
    (0, 0.0, -0.3, 0.7),  # south of obstacle 1, before gate 1 (pathplanner_cs: -0.2)
    (1, 0.25, -0.3, 1.0),  # east of obstacle 2, before gate 2
    (2, 0.1, 0.35, 1.3),  # north of obstacle 3, U-turn behind gate 3, high enough to clear its top
    (3, 0.05, 0.22, 1.2),  # north of obstacle 4, before gate 4
    (1, 0.25, -0.3, 1.2),  # east of obstacle 2, before gate 2 (second visit)
]
# Tangent magnitude at each gate visit (None = automatic Catmull-Rom tangent, no pinning)
# Gate 3: free exit tangent (tight turn around obstacle 3); gate 4: straighter entry past obstacle 4
GATE_TANGENTS: list[float | None] = [1.0, 1.0, None, 1.6, 1.0]


def obstacle_avoid_points(obstacles_pos: NDArray) -> NDArray:
    """Avoidance points anchored to the (observed) obstacle positions."""
    o = obstacles_pos
    return np.array([[o[i][0] + dx, o[i][1] + dy, z] for i, dx, dy, z in AVOID])


class StatePPRacer(Controller):
    """pathplanner_cs waypoints + time-optimal speed profile + full-state feed-forward."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialise the controller and plan the initial path.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        # Thrust model of the onboard Mellinger controller vs. the simulated drone
        dyn = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        mel = load_fn_params(state2attitude, config.sim.drone)
        pwm_gain = float(mel["mass_thrust"]) * 4 * mel["thrust_max"] / mel["pwm_max"]  # ~1.61
        # realised force = pwm_gain * m_ctrl * (a_cmd + g)  ->  a_cmd + g = scale * (a_ref + g)
        self._scale0 = float(dyn["mass"]) / (pwm_gain * float(mel["mass"])) if THRUST_COMP else 1.0
        self._scale = self._scale0

        self._start_pos = np.asarray(obs["pos"], dtype=float).copy()
        self._read_track(obs)
        self._build_plan()
        self._s = 0.0
        self._tick = 0
        self._finished = False

    # -----------------------------------------------------------------------------------------
    # Path
    # -----------------------------------------------------------------------------------------
    def _read_track(self, obs: dict[str, NDArray[np.floating]]):
        self._gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()
        self._gates_seen = np.asarray(obs["gates_visited"], dtype=bool).copy()

    def _track_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        return bool(
            np.max(np.abs(obs["gates_pos"] - self._gates_pos)) > POSE_TOL
            or np.max(np.abs(obs["gates_quat"] - self._gates_quat)) > POSE_TOL
            or np.max(np.abs(obs["obstacles_pos"] - self._obstacles_pos)) > POSE_TOL
            or np.any(np.asarray(obs["gates_visited"], dtype=bool) != self._gates_seen)
        )

    def _gate_normal(self, gate: int, ref_dir: NDArray) -> NDArray:
        """Horizontal gate normal, oriented along the travel direction ref_dir."""
        n = R.from_quat(self._gates_quat[gate]).apply([1.0, 0.0, 0.0])
        n[2] = 0.0
        n /= np.linalg.norm(n) + 1e-9
        return n if n @ ref_dir >= 0 else -n

    def _waypoints(self) -> tuple[NDArray, NDArray]:
        """Waypoint list of pathplanner_cs plus tangents (NaN = automatic)."""
        g = self._gates_pos.copy()
        g[:, 2] += GATE_Z_OFFSET
        av = obstacle_avoid_points(self._obstacles_pos)
        sp = SUPPORT_POINTS
        takeoff = self._start_pos + np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        # (point, gate index or -1)
        seq = [
            (self._start_pos, -1),
            (takeoff, -1),
            (av[0], -1),
            (g[0], 0),
            (av[1], -1),
            (g[1], 1),
            (sp[0], -1),
            (sp[1], -1),
            (g[2], 2),
            (av[2], -1),
            (av[3], -1),
            (g[3], 3),
            (av[4], -1),
            (g[1], 1),
            (sp[2], -1),
        ]
        pts = np.array([p for p, _ in seq], dtype=float)
        tans = np.full_like(pts, np.nan)
        visit = 0
        for i, (_, gate) in enumerate(seq):
            if gate >= 0:
                mag = GATE_TANGENTS[visit]
                visit += 1
                if mag is not None:
                    ref_dir = pts[i + 1] - pts[i - 1]
                    tans[i] = mag * self._gate_normal(gate, ref_dir)
        return pts, tans

    @staticmethod
    def _fit_path(pts: NDArray, tans: NDArray) -> tuple[CubicHermiteSpline, NDArray]:
        """Chord-length parameterised Hermite spline, NaN tangents become Catmull-Rom tangents."""
        u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        u = u + np.arange(len(u)) * 1e-6
        cr = np.empty_like(pts)
        cr[1:-1] = (pts[2:] - pts[:-2]) / (u[2:] - u[:-2])[:, None]
        cr[0] = (pts[1] - pts[0]) / (u[1] - u[0])
        cr[-1] = (pts[-1] - pts[-2]) / (u[-1] - u[-2])
        m = np.where(np.isnan(tans), cr, tans)
        return CubicHermiteSpline(u, pts, m), u

    # -----------------------------------------------------------------------------------------
    # Speed profile
    # -----------------------------------------------------------------------------------------
    @staticmethod
    def _speed_profile(s: NDArray, tang: NDArray, kappa: NDArray, v_cap: NDArray) -> NDArray:
        """Time-optimal v(s) under speed, acceleration and thrust limits (forward/backward)."""
        kn = np.linalg.norm(kappa, axis=1)
        v_lim = np.minimum(v_cap, np.sqrt(A_LAT_MAX / np.maximum(kn, 1e-6)))
        ds = np.diff(s).tolist()
        kn2 = (kn**2).tolist()
        kgz = (2.0 * G * kappa[:, 2]).tolist()
        gt = (G * tang[:, 2]).tolist()
        g2, f2, a2 = G * G, F_MAX**2, A_MAX**2

        def at_bounds(i: int, vi: float) -> tuple[float, float]:
            """Feasible tangential acceleration range at sample i and speed vi."""
            v2 = vi * vi
            c2 = v2 * v2 * kn2[i]  # |v^2 kappa|^2
            w_perp2 = max(c2 + v2 * kgz[i] + g2 - gt[i] ** 2, 0.0)  # |c + g|^2 - (g.T)^2
            r = max(f2 - w_perp2, 0.0) ** 0.5
            r_a = max(a2 - c2, 0.0) ** 0.5
            return min(max(-r - gt[i], -r_a), -0.3), max(min(r - gt[i], r_a), 0.3)

        v = v_lim.tolist()
        v[0] = 0.0
        n = len(v)
        for i in range(n - 1):
            v[i + 1] = min(v[i + 1], (v[i] ** 2 + 2 * at_bounds(i, v[i])[1] * ds[i]) ** 0.5)
        for i in range(n - 2, -1, -1):
            v[i] = min(v[i], (v[i + 1] ** 2 - 2 * at_bounds(i + 1, v[i + 1])[0] * ds[i]) ** 0.5)
        return np.array(v)

    def _build_plan(self):
        pts, tans = self._waypoints()
        spl, u = self._fit_path(pts, tans)
        us = np.linspace(0.0, u[-1], max(int(u[-1] / DS), 10))
        p = spl(us)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        d1, d2 = spl(us, 1), spl(us, 2)
        sp_u = np.linalg.norm(d1, axis=1) + 1e-9
        tang = d1 / sp_u[:, None]
        kappa = (d2 - np.sum(d2 * tang, axis=1)[:, None] * tang) / sp_u[:, None] ** 2
        # Slow down near gates whose true pose is still unknown (revealed within 0.7 m)
        v_cap = np.full(len(s), V_MAX)
        for gate, seen in zip(self._gates_pos, self._gates_seen):
            if not seen:
                near = np.linalg.norm(p - gate, axis=1) < UNSEEN_R
                v_cap[near] = np.minimum(v_cap[near], V_UNSEEN)
        v = self._speed_profile(s, tang, kappa, v_cap)
        a_t = np.zeros_like(v)
        a_t[:-1] = (v[1:] ** 2 - v[:-1] ** 2) / (2 * np.maximum(np.diff(s), 1e-6))
        self._wps = pts
        self._path_s, self._path_p, self._path_v, self._path_t = s, p, v, tang
        self._path_vel = v[:, None] * tang
        self._path_acc = a_t[:, None] * tang + (v**2)[:, None] * kappa
        self._path_len = float(s[-1])
        # Planned lap time (for information)
        self._t_plan = float(np.sum(np.diff(s) / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)))

    def _reproject(self, p_old: NDArray, s_old: float) -> float:
        idx = np.flatnonzero(np.abs(self._path_s - s_old) < 0.6)
        if len(idx) == 0:
            return min(s_old, self._path_len)
        k = idx[np.argmin(np.linalg.norm(self._path_p[idx] - p_old, axis=1))]
        return float(self._path_s[k])

    def _reference(self, s: float) -> tuple[NDArray, NDArray, NDArray, float]:
        """Reference position, velocity, acceleration and speed at progress s."""
        if s >= self._path_len:
            pe = self._path_p[-1] + self._path_t[-1] * (s - self._path_len)
            return pe, self._path_vel[-1].copy(), np.zeros(3), float(self._path_v[-1])
        i = int(np.searchsorted(self._path_s, s, side="right")) - 1
        i = max(min(i, len(self._path_s) - 2), 0)
        w = (s - self._path_s[i]) / max(self._path_s[i + 1] - self._path_s[i], 1e-9)

        def lerp(x: NDArray) -> NDArray:
            return (1 - w) * x[i] + w * x[i + 1]

        return (
            lerp(self._path_p),
            lerp(self._path_vel),
            lerp(self._path_acc),
            float(lerp(self._path_v)),
        )

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
        if self._track_changed(obs) and not self._finished:
            p_old = self._reference(self._s)[0]
            self._read_track(obs)
            self._build_plan()
            self._s = self._reproject(p_old, self._s)

        v_now = self._reference(self._s)[3]
        pos, vel, _, _ = self._reference(self._s + v_now * POS_LOOKAHEAD)
        acc = self._reference(self._s + v_now * FF_LOOKAHEAD)[2]
        # Extra feedback through the acceleration channel (Mellinger gains are fixed)
        acc = acc + KP_EXTRA * (pos - obs["pos"]) + KD_EXTRA * (vel - obs["vel"])
        # Invert the thrust scaling of the onboard controller
        g = np.array([0.0, 0.0, G])
        acc_cmd = self._scale * (acc + g) - g
        self._last_ref = pos
        return np.concatenate(
            (pos, vel, acc_cmd, [0.0, 0.0, 0.0, 1.0], np.zeros(3)), dtype=np.float32
        )

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the progress variable and adapt the thrust scale.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        p_ref, vel_ref, _, v = self._reference(self._s)
        # Thrust scale adaptation from the steady height error (unknown mass in level 2)
        if SCALE_ADAPT > 0:
            e_z = float(p_ref[2] - obs["pos"][2])
            self._scale = float(
                np.clip(
                    self._scale + SCALE_ADAPT * e_z / self._freq,
                    0.8 * self._scale0,
                    1.25 * self._scale0,
                )
            )
        lag = 0.0
        if v > 0.1:
            lag = float((p_ref - obs["pos"]) @ (vel_ref / np.linalg.norm(vel_ref)))
        rate = float(np.clip(1.0 - (lag - LAG_FREE) / LAG_SCALE, RATE_MIN, 1.0))
        dt = rate / self._freq
        v_mid = self._reference(self._s + 0.5 * max(v, V_START) * dt)[3]
        self._s += max(v_mid, V_START) * dt
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._s = 0.0
        self._tick = 0
        self._scale = self._scale0
        self._finished = False

    def render_callback(self, sim: Sim):
        """Visualise the planned path, the waypoints and the current setpoint."""
        draw_points(sim, self._reference(self._s)[0].reshape(1, -1), rgba=(1, 0, 0, 1), size=0.02)
        draw_points(sim, self._wps, rgba=(0.0, 0.0, 1.0, 1.0), size=0.015)
        draw_line(sim, self._path_p[::5], rgba=(0.0, 1.0, 0.0, 1.0))
