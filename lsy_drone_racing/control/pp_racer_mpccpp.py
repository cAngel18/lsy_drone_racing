"""MPCC++-style racing controller for the Crazyflie attitude interface (self-contained).

After Romero et al. (T-RO 2022) and Krinner et al. (MPCC++, RSS 2024), adapted to the attitude
interface and late gate reveals.

**Planner.** Chord-length Hermite path through the gates, obstacle avoidance and support points
(racing line optimised offline for minimum planned time), sampled every ``DS`` in arc length and
rebuilt whenever a gate or obstacle pose is revealed (progress re-projected onto the new path).

**Model.** Attitude/thrust response identified on this simulator: roll/pitch follow the raw
command as a 2nd-order system, the collective thrust with a ~25 ms lag, plus linear drag. The
thrust is scaled with an online least-squares mass estimate; an acceleration disturbance
estimated from the one-step prediction error enters the model as a parameter (offset-free MPC).

**MPCC++ formulation** (acados, SQP-RTI):

* **Commands are states, their rates the inputs** (Romero: rotor thrusts as states with thrust rate
  inputs): roll/pitch and collective-thrust commands are integrated from penalised rates, so
  bang-bang command chattering is expensive.
* **Progress acceleration is the input**, v_theta a state (Romero: a direct v_theta input gives
  "very noisy control inputs"); the progress reward is linear, -MU v_theta.
* **Rotor thrust limits.** The identified attitude model gives the angular acceleration the command
  produces; the required torque J ddrpy is mapped through the motor mixing onto the four rotors,
  and every rotor must stay within [f_min, f_max] (8 soft linear constraints). This models the
  collective-thrust loss under heavy attitude activity that made the fast MPCC sink at gates.
* **Tunnel constraint (MPCC++)**: square cross-section around the path (normal n, binormal b),
  |e.n| <= W, |e.b| <= H, narrow at gates, pole-limited laterally, softened with the MPCC++ barrier
  log(1 + exp(-ALPHA h)) inside a convex-over-nonlinear cost; the contour weight is small.
* Speed caps before gate/pole reveals (level 2 reveals at 0.7 m; MPCC++ assumes a known track).

Status (level 2, 48 paired runs, tuned weights): 31/48 finished, mean lap 5.96 s, best 5.38 s;
nominal track 5.70 s. See experiments.md.

Requires ``control_mode = "attitude"``. The acados solver is generated once per model/weights hash
into ``c_generated_code/pp_racer_mpccpp_<hash>`` of the repository (independent of the cwd).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.dynamics import load_params as load_dynamics_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates, cs_rpy2matrix
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

G = 9.81  # [m/s^2]

# ---------------------------------------------------------------------------------------------
# Racing line (waypoints optimised offline for minimum planned time, clearance to poles and gate
# frames no worse than the pathplanner_cs line on the nominal and 8 randomised tracks)
# ---------------------------------------------------------------------------------------------
SUPPORT_POINTS = np.array([[0.485, 0.75, 1.155], [-0.443, -0.037, 0.699], [0.5, 1.25, 1.2]])
# Obstacle avoidance points: (obstacle index, dx, dy, absolute z). Order matters.
AVOID = [
    (0, -0.382, -0.234, 0.464),  # south-west of obstacle 1, low line into gate 1
    (1, 0.229, -0.285, 0.915),  # east of obstacle 2, before gate 2
    (2, 0.118, 0.265, 1.279),  # north of obstacle 3, U-turn over gate 3
    (3, 0.045, 0.213, 1.201),  # north of obstacle 4, before gate 4
    (1, 0.137, -0.302, 1.205),  # east of obstacle 2, before gate 2 (second visit)
]
GATE_TANGENTS: list[float | None] = [0.988, 1.213, None, 1.143, 1.573]  # None = Catmull-Rom
DS = 0.02  # [m] arc length resolution of the sampled path
POSE_TOL = 1e-3  # change of an observed pose that triggers a re-plan
GATE_SIGMA = 0.3  # [m] arc-length width of the gate bump on the approach side
GATE_SIGMA_POST = 0.08  # [m] arc-length width behind the gate plane

# ---------------------------------------------------------------------------------------------
# Identified model (this simulator)
# ---------------------------------------------------------------------------------------------
# Attitude response to the raw command: ddrpy = A rpy + B drpy + C cmd
ATT_A = [-163.57, -177.33, -144.72]
ATT_B = [-25.36, -27.02, -17.37]
ATT_C = [151.83, 172.75, 127.01]  # yaw: crazyflow so_rpy values (yaw is held at 0)
THRUST_TAU = 0.025  # [s] first-order lag of the collective thrust
THRUST_GAIN = 0.98  # realised / commanded thrust
# Online mass identification (level 2 randomises the mass by +-0.005 kg = +-11.5 %)
MASS_EST = True
MASS_PRIOR_N = 25.0  # weight of the nominal mass prior, in equivalent hover samples
MASS_FORGET = 0.995  # forgetting factor per step of the least-squares mass estimate
MASS_BOUNDS = (0.85, 1.15)  # bounds of the estimate relative to the nominal mass

# ---------------------------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------------------------
N_HORIZON = 40  # stages
DT = 0.02  # [s]
# Cost (convex over nonlinear): 0.5 sum W r^2 + B_TUNNEL sum softplus - MU v_theta
# MU, Q_CONTOUR, B_TUNNEL, R_DCMD, R_DTHRUST, W_GATE, H_FREE, CAP_V_UNSEEN: Bayesian-optimised
# (TuRBO-style, 30 x 24 paired level-2 episodes, objective = expected lap time, crash = 10 s)
MU = 11.8  # progress reward per stage [1/(m/s)]
Q_CONTOUR = 24.0  # small (shape only), the tunnel does the safety
Q_LAG = 1500.0  # keeps the progress variable at the drone
Q_DRPY = 0.05
R_DCMD = 0.0128  # roll/pitch command rate [(rad/s)^-2]
R_DTHRUST = 0.49  # thrust command rate [(N/s)^-2]
R_ATHETA = 0.01  # progress acceleration [(m/s^2)^-2]
R_YAW = 5.0
B_TUNNEL = 2.5  # tunnel barrier weight
ALPHA = 100.0  # barrier sharpness [1/m] (MPCC++)
# Tunnel half widths
W_FREE = 0.25  # [m] lateral, away from gates and poles
H_FREE = 0.19  # [m] vertical, away from gates
W_GATE = 0.065  # [m] at gate openings (opening +-0.2 m, drone half size ~0.09 m)
POLE_MARGIN = 0.14  # [m] lateral tunnel edge distance to pole axes
W_MIN = 0.05
# Limits
MAX_TILT_CMD = 0.8  # [rad]
LOW_Z = 0.08  # [m] below this height the tilt command is limited to LOW_TILT (take-off)
LOW_TILT = 0.2  # [rad]
DCMD_MAX = 20.0  # [rad/s] roll/pitch command rate
DTHRUST_MAX = 20.0  # [N/s]
ATHETA_MAX = 30.0  # [m/s^2]
THRUST_FRAC = 0.95  # collective command bound (the rotor constraints do the rest)
ROTOR_FRAC = 0.97  # usable fraction of the per-rotor thrust range
V_THETA_MAX = 5.0
V_THETA_MIN_GUESS = 0.3  # [m/s] progress speed of the first guess
# Soft constraint penalties: [v_theta cap, speed cap, 8 rotor limits]
SLACK_V = (1e3, 1e4)  # L1 [1/(m/s)], L2 [1/(m/s)^2]
SLACK_ROTOR = (2e3, 2e5)  # L1 [1/N], L2 [1/N^2]
# Speed caps (loose time-optimal profile + reveal caps)
CAP_V_MAX = 4.0
CAP_A_LAT = 11.0  # [m/s^2] centripetal acceleration
CAP_A_MAX = 11.0  # [m/s^2] total acceleration (tangential + centripetal)
CAP_F_MAX = 15.0  # [m/s^2] mass-normalised thrust |a + g|
CAP_V_UNSEEN = 2.42  # [m/s] within CAP_R_GATE of gates not revealed yet
CAP_R_GATE = 1.0
CAP_V_OBS = 2.5  # [m/s] within CAP_R_OBS of poles not revealed yet
CAP_R_OBS = 0.6
# Disturbance observer (offset-free MPC)
DIST_GAIN = 0.05  # observer gain of the acceleration disturbance (per step, 0 = off)
DIST_MAX = 1.5  # [m/s^2] bound of the disturbance estimate
# Solver / progress
QP_ITER_MAX = 50
ITERS_COLD = 5  # RTI iterations on a cold start / after a solver failure
ITERS_REPLAN = 3  # RTI iterations right after a replan
PROJ_BACK, PROJ_FWD = 0.1, 0.3  # [m] search window of the progress projection (no skipping)

_OK_STATUS = (0, 2)  # acados statuses with a usable solution (2 = QP max iterations)
_MODEL_VERSION = 2
_OCP_KEYS = (
    "N_HORIZON", "DT", "MU", "Q_CONTOUR", "Q_LAG", "Q_DRPY", "R_DCMD", "R_DTHRUST", "R_ATHETA",
    "R_YAW", "B_TUNNEL", "ALPHA", "MAX_TILT_CMD", "DCMD_MAX", "DTHRUST_MAX", "ATHETA_MAX",
    "THRUST_FRAC", "ROTOR_FRAC", "V_THETA_MAX", "SLACK_V", "SLACK_ROTOR", "QP_ITER_MAX",
)  # fmt: skip
_SOLVERS: dict[str, AcadosOcpSolver] = {}  # one solver per configuration and process
NX, NU, NP = 18, 4, 20
# State: pos 0:3, vel 3:6, rpy 6:9, drpy 9:12, thrust 12, cmd_roll 13, cmd_pitch 14, cmd_thrust 15,
#        theta 16, v_theta 17.  Input: d cmd_roll, d cmd_pitch, d cmd_thrust, a_theta.
# Parameters: p_ref 0:3, tangent 3:6, normal 6:9, binormal 9:12, theta_hat 12, W 13, H 14,
#             v_cap 15, d_acc 16:19, thrust scale m_hat/m_nom 19.


def _softplus(z: cs.SX) -> cs.SX:
    """Numerically safe log(1 + exp(z)) (no overflow far outside the tunnel)."""
    return cs.fmax(z, 0) + cs.log(1 + cs.exp(-cs.fabs(z)))


def _build_solver(dyn: dict, m: float, kd: NDArray, t_min: float, t_max: float) -> AcadosOcpSolver:
    """Create (or load) the MPCC++ solver."""
    cfg = {k: globals()[k] for k in _OCP_KEYS}
    cfg.update(
        v=_MODEL_VERSION, m=m, kd=list(map(float, kd)), t=[t_min, t_max],
        att=[ATT_A, ATT_B, ATT_C, THRUST_TAU, THRUST_GAIN],
    )  # fmt: skip
    key = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    if key in _SOLVERS:
        return _SOLVERS[key]
    x = cs.SX.sym("x", NX)
    u = cs.SX.sym("u", NU)
    par = cs.SX.sym("par", NP)
    pos, vel, rpy, drpy, thrust = x[0:3], x[3:6], x[6:9], x[9:12], x[12]
    cmd_rp, cmd_t, theta, v_theta = x[13:15], x[15], x[16], x[17]
    p_ref, tang, nrm, bnm = par[0:3], par[3:6], par[6:9], par[9:12]
    theta_hat, w_lat, h_vert, v_cap, d_acc, t_scale = (
        par[12], par[13], par[14], par[15], par[16:19], par[19]
    )  # fmt: skip

    cmd_rpy = cs.vertcat(cmd_rp, 0.0)  # yaw command 0
    z_body = cs_rpy2matrix(rpy)[:, 2]
    acc = z_body * THRUST_GAIN * thrust / m - cs.DM(kd) * vel - cs.DM([0, 0, G]) + d_acc
    ddrpy = cs.DM(ATT_A) * rpy + cs.DM(ATT_B) * drpy + cs.DM(ATT_C) * cmd_rpy
    f = cs.vertcat(vel, acc, drpy, ddrpy, (cmd_t - thrust) / THRUST_TAU, u[0:3], v_theta, u[3])

    # Rotor limits: torque needed for the predicted angular acceleration, mapped onto the rotors
    J = np.diag(np.asarray(dyn["J"], dtype=float))
    arm = float(dyn["L"])
    mix = np.asarray(dyn["mixing_matrix"], dtype=float)  # rows: roll, pitch, yaw torque signs
    delta = cs.vertcat(J[0] * ddrpy[0], J[1] * ddrpy[1]) / (4.0 * arm)  # [N] per rotor
    f_mean = THRUST_GAIN * thrust * t_scale / 4.0
    f_lo, f_hi = float(dyn["thrust_min"]) * 1.05, float(dyn["thrust_max"]) * ROTOR_FRAC
    rotor = [f_mean + mix[0, i] * delta[0] + mix[1, i] * delta[1] for i in range(4)]
    h_rotor = cs.vertcat(*[ri - f_hi for ri in rotor], *[f_lo - ri for ri in rotor])

    # Contouring errors (path linearised around theta_hat) and tunnel distances
    e = pos - (p_ref + tang * (theta - theta_hat))
    lag = cs.dot(tang, e)
    contour = e - tang * lag
    e_n, e_b = cs.dot(nrm, e), cs.dot(bnm, e)
    h_tunnel = cs.vertcat(w_lat - e_n, w_lat + e_n, h_vert - e_b, h_vert + e_b)  # >= 0 inside

    model = AcadosModel()
    model.name = f"pp_racer_mpccpp_{key}"
    model.x, model.u, model.p = x, u, par
    model.f_expl_expr = f

    # Convex-over-nonlinear cost: psi(y) = 0.5 y_q' W y_q + B sum softplus(-ALPHA y_h) - MU y_v
    y_q = cs.vertcat(contour, lag, drpy, u, rpy[2])  # 3 + 1 + 3 + 4 + 1 = 12
    w_q = [Q_CONTOUR] * 3 + [Q_LAG] + [Q_DRPY] * 3 + [R_DCMD] * 2 + [R_DTHRUST, R_ATHETA, R_YAW]
    model.cost_y_expr = cs.vertcat(y_q, h_tunnel, v_theta)
    r = cs.SX.sym("r", 12 + 4 + 1)
    sp = _softplus(-ALPHA * r[12:16])
    model.cost_r_in_psi_expr = r
    model.cost_psi_expr = (
        0.5 * cs.sum1(cs.DM(w_q) * r[0:12] ** 2) + B_TUNNEL * cs.sum1(sp) - MU * r[16]
    )
    y_qe = cs.vertcat(contour, lag, drpy)
    w_qe = [Q_CONTOUR] * 3 + [2 * Q_LAG] + [Q_DRPY] * 3
    model.cost_y_expr_e = cs.vertcat(y_qe, h_tunnel)
    re = cs.SX.sym("re", 7 + 4)
    model.cost_r_in_psi_expr_e = re
    model.cost_psi_expr_e = 0.5 * cs.sum1(cs.DM(w_qe) * re[0:7] ** 2) + B_TUNNEL * cs.sum1(
        _softplus(-ALPHA * re[7:11])
    )

    model.con_h_expr = cs.vertcat(v_theta - v_cap, cs.dot(vel, tang) - v_cap, h_rotor)  # <= 0

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf = N_HORIZON * DT
    ocp.cost.cost_type = ocp.cost.cost_type_e = "CONVEX_OVER_NONLINEAR"
    ocp.cost.yref = np.zeros(17)
    ocp.cost.yref_e = np.zeros(11)
    # Hard bounds: command states, progress speed; input rates
    ocp.constraints.idxbx = np.array([13, 14, 15, 17])
    ocp.constraints.lbx = np.array([-MAX_TILT_CMD, -MAX_TILT_CMD, t_min, 0.0])
    ocp.constraints.ubx = np.array([MAX_TILT_CMD, MAX_TILT_CMD, THRUST_FRAC * t_max, V_THETA_MAX])
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = np.array([-DCMD_MAX, -DCMD_MAX, -DTHRUST_MAX, -ATHETA_MAX])
    ocp.constraints.ubu = np.array([DCMD_MAX, DCMD_MAX, DTHRUST_MAX, ATHETA_MAX])
    ocp.constraints.x0 = np.zeros(NX)
    nh = 10
    ocp.constraints.lh = np.full(nh, -1e3)
    ocp.constraints.uh = np.zeros(nh)
    ocp.constraints.idxsh = np.arange(nh)
    l1 = np.array([SLACK_V[0]] * 2 + [SLACK_ROTOR[0]] * 8)
    l2 = np.array([SLACK_V[1]] * 2 + [SLACK_ROTOR[1]] * 8)
    ocp.cost.zl, ocp.cost.Zl = np.zeros(nh), np.zeros(nh)
    ocp.cost.zu, ocp.cost.Zu = l1, l2
    par0 = np.zeros(NP)
    par0[3], par0[6], par0[11] = 1.0, 1.0, 1.0
    par0[13:16] = [0.2, 0.2, 1.0]
    par0[19] = 1.0
    ocp.parameter_values = par0

    opts = ocp.solver_options
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    opts.qp_solver_cond_N = max(N_HORIZON // 4, 1)
    opts.hessian_approx = "GAUSS_NEWTON"
    opts.integrator_type = "ERK"
    opts.sim_method_num_stages = 4
    opts.sim_method_num_steps = 2
    opts.nlp_solver_type = "SQP_RTI"
    opts.qp_solver_warm_start = 1
    opts.qp_solver_iter_max = QP_ITER_MAX
    opts.hpipm_mode = "ROBUST"

    out = Path(__file__).resolve().parents[2] / "c_generated_code" / model.name
    out.mkdir(parents=True, exist_ok=True)
    ocp.code_export_directory = str(out)
    with open(out.parent / f"{model.name}.lock", "w") as lock:  # parallel processes: build once
        fcntl.flock(lock, fcntl.LOCK_EX)
        built = (out / f"libacados_ocp_solver_{model.name}.so").exists()
        solver = AcadosOcpSolver(
            ocp,
            json_file=str(out / f"{model.name}.json"),
            verbose=False,
            generate=not built,
            build=not built,
        )
    _SOLVERS[key] = solver
    return solver


class PPRacerMPCCPP(Controller):
    """MPCC++-style racing MPCC with rotor limits, command-rate inputs and a tunnel."""

    _THETA = 16  # progress variable in the state (see NX layout)

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the path and set up the solver.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        dyn = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        self._m_nom = float(dyn["mass"])
        self._drag = np.asarray(dyn.get("drag_matrix", np.zeros((3, 3))), dtype=float)  # body
        self._t_max = 4.0 * float(dyn["thrust_max"])  # [N]
        self._t_min = 4.0 * float(dyn["thrust_min"])  # [N]
        self._reset_estimator()
        self._tick = 0
        self._finished = False

        self._start_pos = np.asarray(obs["pos"], dtype=float).copy()
        self._read_track(obs)
        self._build_plan()

        self._kd = -np.diag(self._drag) / self._m_nom
        self._solver = _build_solver(dyn, self._m_nom, self._kd, self._t_min, self._t_max)
        self._reset_mpc_state()

    def _reset_mpc_state(self):
        self._theta = 0.0
        self._X = None  # last state trajectory (N+1, NX)
        self._U = None  # last input trajectory (N, NU)
        self._n_iter = ITERS_COLD
        self._t_act = self._m_nom * G  # estimate of the realised thrust (model state)
        self._u_prev = np.array([0.0, 0.0, 0.0, self._m_nom * G])
        self._fails = 0
        self._d = np.zeros(3)  # acceleration disturbance estimate [m/s^2]
        self._v_pred = None  # model prediction of the velocity after the current step
        self._cmd = np.array([0.0, 0.0, self._m_nom * G])  # roll, pitch, thrust (nominal units)
        self._v_theta = 0.0

    # -----------------------------------------------------------------------------------------
    # Online mass estimate
    # -----------------------------------------------------------------------------------------
    def _reset_estimator(self):
        self._m_hat = self._m_nom
        w0 = MASS_PRIOR_N * G * G
        self._ls_num, self._ls_den = w0 * self._m_nom, w0  # sum(T*y), sum(y*y)
        self._pending = None  # (thrust [N], velocity) of the last command

    def _drag_force(self, quat: NDArray, vel: NDArray) -> NDArray:
        """Aerodynamic drag force [N] (world frame) at velocity vel."""
        rot = R.from_quat(quat).as_matrix()
        return rot @ (self._drag @ (rot.T @ np.asarray(vel, dtype=float)))

    def _update_mass(self, obs: dict):
        """Least-squares update of m in  T = m * (a_meas + g) . z_body  (with forgetting)."""
        if self._pending is None:
            return
        thrust, vel0 = self._pending
        self._pending = None
        if obs["pos"][2] < 0.08 or not (0.15 * self._t_max < thrust < 0.85 * self._t_max):
            return
        z_body = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        a_meas = (np.asarray(obs["vel"]) - vel0) * self._freq
        y = float((a_meas + np.array([0.0, 0.0, G])) @ z_body)
        thrust = thrust + float(self._drag_force(obs["quat"], vel0) @ z_body)
        self._ls_num = MASS_FORGET * self._ls_num + thrust * y
        self._ls_den = MASS_FORGET * self._ls_den + y * y
        lo, hi = MASS_BOUNDS
        self._m_hat = float(
            np.clip(self._ls_num / self._ls_den, lo * self._m_nom, hi * self._m_nom)
        )

    # -----------------------------------------------------------------------------------------
    # Track observation
    # -----------------------------------------------------------------------------------------
    def _read_track(self, obs: dict[str, NDArray[np.floating]]):
        self._gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()
        self._gates_seen = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._obst_seen = np.asarray(obs["obstacles_visited"], dtype=bool).copy()

    def _track_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        return bool(
            np.max(np.abs(obs["gates_pos"] - self._gates_pos)) > POSE_TOL
            or np.max(np.abs(obs["gates_quat"] - self._gates_quat)) > POSE_TOL
            or np.max(np.abs(obs["obstacles_pos"] - self._obstacles_pos)) > POSE_TOL
            or np.any(np.asarray(obs["gates_visited"], dtype=bool) != self._gates_seen)
            or np.any(np.asarray(obs["obstacles_visited"], dtype=bool) != self._obst_seen)
        )

    # -----------------------------------------------------------------------------------------
    # Path
    # -----------------------------------------------------------------------------------------
    def _gate_normal(self, gate: int, ref_dir: NDArray) -> NDArray:
        """Horizontal gate normal, oriented along the travel direction ref_dir."""
        n = R.from_quat(self._gates_quat[gate]).apply([1.0, 0.0, 0.0])
        n[2] = 0.0
        n /= np.linalg.norm(n) + 1e-9
        return n if n @ ref_dir >= 0 else -n

    def _waypoints(self) -> tuple[NDArray, NDArray]:
        """Waypoints of the racing line plus tangents (NaN = automatic)."""
        g = self._gates_pos
        o = self._obstacles_pos
        av = np.array([[o[i][0] + dx, o[i][1] + dy, z] for i, dx, dy, z in AVOID])
        sp = SUPPORT_POINTS
        # (point, gate index or -1); gate 3 -> 4: loop north around obstacle 3, over gate 3
        seq = [
            (self._start_pos, -1),
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
                    tans[i] = mag * self._gate_normal(gate, pts[i + 1] - pts[i - 1])
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

    def _speed_profile(self, s: NDArray, tang: NDArray, kappa: NDArray, v_cap: NDArray) -> NDArray:
        """Time-optimal v(s) under speed, acceleration and thrust limits (forward/backward)."""
        kn = np.linalg.norm(kappa, axis=1)
        v_lim = np.minimum(v_cap, np.sqrt(CAP_A_LAT / np.maximum(kn, 1e-6)))
        ds = np.diff(s).tolist()
        kn2 = (kn**2).tolist()
        kgz = (2.0 * G * kappa[:, 2]).tolist()
        gt = (G * tang[:, 2]).tolist()
        g2, f2, a2 = G * G, CAP_F_MAX**2, CAP_A_MAX**2

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
        """Sampled path + gate bump + tunnel half widths W(s), H(s) + loose speed cap v_cap(s)."""
        pts, tans = self._waypoints()
        spl, u = self._fit_path(pts, tans)
        us = np.linspace(0.0, u[-1], max(int(u[-1] / DS), 10))
        p = spl(us)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        d1 = spl(us, 1)
        tang = d1 / (np.linalg.norm(d1, axis=1) + 1e-9)[:, None]
        self._wps = pts
        self._path_s, self._path_p, self._path_t = s, p, tang
        self._path_len = float(s[-1])

        # Gate bump 0..1: rises on the approach to every gate pass (arc length s_g, where the path
        # goes through a gate centre) and drops quickly behind the gate plane, so that tight turns
        # right after a gate (U-turn behind gate 3) are not narrowed.
        bump = np.zeros_like(s)
        for gate in self._gates_pos:
            d = np.linalg.norm(p - gate, axis=1)
            near = d < 0.06
            # one pass per contiguous run of samples close to the gate centre
            edges = np.flatnonzero(np.diff(np.concatenate([[0], near.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                s_g = s[a + int(np.argmin(d[a:b]))]
                sig = np.where(s <= s_g, GATE_SIGMA, GATE_SIGMA_POST)
                bump = np.maximum(bump, np.exp(-0.5 * ((s - s_g) / sig) ** 2))
        self._path_bump = bump

        # Tunnel: lateral width limited by the poles, both widths narrowed at gate openings
        d_pole = np.min(
            np.linalg.norm(p[:, None, :2] - self._obstacles_pos[None, :, :2], axis=2), axis=1
        )
        w = np.clip(d_pole - POLE_MARGIN, W_MIN, W_FREE)
        self._path_w = w - (w - np.minimum(w, W_GATE)) * bump
        self._path_h = H_FREE - (H_FREE - W_GATE) * bump

        # Speed cap: loose time-optimal profile, slower near gates/poles not revealed yet
        kappa = np.gradient(tang, s, axis=0)
        kappa -= np.sum(kappa * tang, axis=1)[:, None] * tang
        v_cap = np.full(len(s), CAP_V_MAX)
        for gate, seen in zip(self._gates_pos, self._gates_seen):
            if not seen:
                near = np.linalg.norm(p - gate, axis=1) < CAP_R_GATE
                v_cap[near] = np.minimum(v_cap[near], CAP_V_UNSEEN)
        for obst, seen in zip(self._obstacles_pos, self._obst_seen):
            if not seen:
                near = np.linalg.norm(p[:, :2] - obst[:2], axis=1) < CAP_R_OBS
                v_cap[near] = np.minimum(v_cap[near], CAP_V_OBS)
        self._path_vcap = np.maximum(self._speed_profile(s, tang, kappa, v_cap), 0.3)

    def _reproject(self, p_old: NDArray, s_old: float) -> float:
        """Progress on the new path closest to the old reference point (after a replan)."""
        idx = np.flatnonzero(np.abs(self._path_s - s_old) < 0.6)
        if len(idx) == 0:
            return min(s_old, self._path_len)
        k = idx[np.argmin(np.linalg.norm(self._path_p[idx] - p_old, axis=1))]
        return float(self._path_s[k])

    def _path_at(self, theta: NDArray) -> tuple[NDArray, NDArray]:
        """Path point and unit tangent at arc lengths theta (straight beyond the end)."""
        s = self._path_s
        th = np.clip(theta, 0.0, s[-1])
        p = np.column_stack([np.interp(th, s, self._path_p[:, i]) for i in range(3)])
        t = np.column_stack([np.interp(th, s, self._path_t[:, i]) for i in range(3)])
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        p = p + t * (theta - th)[:, None]
        return p, t

    def _frame_at(self, theta: NDArray) -> tuple[NDArray, ...]:
        """Path point, tangent, normal (horizontal), binormal, W, H and v_cap at theta."""
        p, t = self._path_at(theta)
        n = np.cross(np.array([0.0, 0.0, 1.0]), t)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(nn > 1e-3, n / np.maximum(nn, 1e-9), np.array([1.0, 0.0, 0.0]))
        b = np.cross(t, n)
        th = np.clip(theta, 0.0, self._path_s[-1])
        w = np.interp(th, self._path_s, self._path_w)
        h = np.interp(th, self._path_s, self._path_h)
        vc = np.interp(th, self._path_s, self._path_vcap)
        return p, t, n, b, w, h, vc

    def _project(self, pos: NDArray) -> float:
        """Closest path point, searched around the progress the MPC predicted for this step."""
        centre = (
            self._theta if self._X is None else max(float(self._X[1, self._THETA]), self._theta)
        )
        last = len(self._path_s) - 1
        lo = min(int(np.searchsorted(self._path_s, centre - PROJ_BACK)), last)
        hi = max(int(np.searchsorted(self._path_s, centre + PROJ_FWD)), lo + 1)
        k = lo + int(np.argmin(np.sum((self._path_p[lo:hi] - pos) ** 2, axis=1)))
        return float(self._path_s[min(k, last)])

    # -----------------------------------------------------------------------------------------
    # MPCC step
    # -----------------------------------------------------------------------------------------
    def _mpcc(self, obs: dict) -> NDArray:
        n, solver = N_HORIZON, self._solver
        pos = np.asarray(obs["pos"], dtype=float)
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = np.asarray(ang_vel2rpy_rates(obs["quat"], obs["ang_vel"]), dtype=float)
        self._theta = self._project(pos)
        if self._X is not None:
            self._v_theta = float(self._X[1, 17])
        x0 = np.concatenate(
            [pos, obs["vel"], rpy, drpy, [self._t_act], self._cmd, [self._theta, self._v_theta]]
        )

        # Progress guess per stage: shifted last solution, else the current speed
        if self._X is None:
            v0 = max(float(np.linalg.norm(obs["vel"])), V_THETA_MIN_GUESS)
            inc = np.full(n, v0 * DT)
        else:
            inc = np.diff(self._X[:, 16])
            inc = np.concatenate([inc[1:], inc[-1:]])
        inc = np.clip(inc, 0.0, V_THETA_MAX * DT)
        theta_hat = self._theta + np.concatenate([[0.0], np.cumsum(inc)])
        p_hat, t_hat, n_hat, b_hat, w_hat, h_hat, vc_hat = self._frame_at(theta_hat)

        if self._X is None:
            xg = np.tile(x0, (n + 1, 1))
            ug = np.zeros((n, NU))
        else:
            xg = np.vstack([self._X[1:], self._X[-1:]])
            ug = np.vstack([self._U[1:], self._U[-1:]])
        xg[:, 16] = theta_hat
        xg[0] = x0
        tilt = LOW_TILT if pos[2] < LOW_Z else MAX_TILT_CMD
        # Thrust bound in model (nominal-mass) units: the command is scaled by m_hat / m_nom
        t_ub = THRUST_FRAC * self._t_max * self._m_nom / self._m_hat
        scale = self._m_hat / self._m_nom
        lbx = np.array([-tilt, -tilt, self._t_min, 0.0])
        ubx = np.array([tilt, tilt, t_ub, V_THETA_MAX])
        for k in range(n + 1):
            par = np.concatenate(
                [
                    p_hat[k], t_hat[k], n_hat[k], b_hat[k],
                    [theta_hat[k], w_hat[k], h_hat[k], vc_hat[k]], self._d, [scale],
                ]
            )  # fmt: skip
            solver.set(k, "p", par)
            solver.set(k, "x", xg[k])
            if k < n:
                solver.set(k, "u", ug[k])
            if 0 < k < n:
                solver.set(k, "lbx", lbx)
                solver.set(k, "ubx", ubx)
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        status = 0
        for _ in range(self._n_iter):
            status = solver.solve()
            if status not in _OK_STATUS:
                break
        self._n_iter = 1
        if status not in _OK_STATUS:
            self._fails += 1
            self._X = self._U = None
            self._n_iter = ITERS_COLD
            solver.reset()
            return self._u_prev.copy()
        self._X = np.array([solver.get(k, "x") for k in range(n + 1)])
        self._U = np.array([solver.get(k, "u") for k in range(n)])
        self._v_pred = (self._X[1, 3:6].copy(), np.asarray(obs["vel"], dtype=float).copy())
        # Command for the coming interval: the command states at the end of the first stage
        self._cmd = self._X[1, 13:16].copy()
        return np.array([self._cmd[0], self._cmd[1], 0.0, self._cmd[2]])

    # -----------------------------------------------------------------------------------------
    # Controller interface
    # -----------------------------------------------------------------------------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the attitude command [roll, pitch, yaw, collective thrust].

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The attitude command [roll, pitch, yaw, thrust].
        """
        if int(obs["n_gates_passed"]) >= len(obs["gate_sequence"]):
            self._finished = True
        if self._track_changed(obs) and not self._finished:
            # Re-map the progress onto the new path (arc lengths shift when a gate moves) and
            # shift the warm start with it; iterate a few times on the changed problem
            p_old = self._path_at(np.array([self._theta]))[0][0]
            self._read_track(obs)
            self._build_plan()
            theta_new = self._reproject(p_old, self._theta)
            if self._X is not None:
                self._X[:, self._THETA] += theta_new - self._theta
            self._theta = theta_new
            self._n_iter = max(self._n_iter, ITERS_REPLAN)
        u = self._mpcc(obs)
        if not np.all(np.isfinite(u)):
            u = self._u_prev.copy()
        self._u_prev = u.copy()
        # Realised thrust (first-order lag, nominal-mass units): its mean over the coming step is
        # what the mass estimator compares with the measured acceleration
        t_before = self._t_act
        self._t_act += (1.0 - np.exp(-DT / THRUST_TAU)) * (u[3] - self._t_act)
        scale = self._m_hat / self._m_nom
        thrust = float(np.clip(u[3] * scale, self._t_min, self._t_max))
        if MASS_EST:
            t_mean = 0.5 * (t_before + self._t_act) * scale * THRUST_GAIN
            self._pending = (t_mean, np.asarray(obs["vel"], dtype=float).copy())
        return np.array([u[0], u[1], u[2], thrust], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Update the mass and disturbance estimates.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        if MASS_EST:
            self._update_mass(obs)
        # Offset-free MPC: low-pass the one-step prediction error of the acceleration
        if self._v_pred is not None and obs["pos"][2] > LOW_Z and DIST_GAIN > 0:
            v_model, _ = self._v_pred
            e = (np.asarray(obs["vel"]) - v_model) * self._freq  # a_meas - a_pred
            self._d = np.clip(self._d + DIST_GAIN * e, -DIST_MAX, DIST_MAX)
        self._v_pred = None
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0
        self._finished = False
        self._reset_estimator()
        self._reset_mpc_state()

    def render_callback(self, sim: Sim):
        """Visualise the planned path, the waypoints and the current progress point."""
        p_now = self._path_at(np.array([self._theta]))[0]
        draw_points(sim, p_now, rgba=(1, 0, 0, 1), size=0.02)
        draw_points(sim, self._wps, rgba=(0.0, 0.0, 1.0, 1.0), size=0.015)
        draw_line(sim, self._path_p[::5], rgba=(0.0, 1.0, 0.0, 1.0))
