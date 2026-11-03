"""MPCC++-style racing controller for the Crazyflie attitude interface.

After Romero et al. (T-RO 2022) and Krinner et al. (MPCC++, RSS 2024), adapted to the attitude
interface and late gate reveals.

Planner, identified attitude/thrust model, progress projection, replanning and mass estimate come
from ``pp_racer_mpcc`` (subclassed). Differences to ``pp_racer_mpcc_fast``:

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
* Speed caps before gate/pole reveals as in ``pp_racer_mpcc_fast`` (level 2 reveals at 0.7 m;
  MPCC++ assumes a known track).

Status (level 2, 48 paired runs, tuned weights): 31/48 finished, mean lap 5.96 s, best 5.38 s;
nominal track 5.70 s. See experiments.md.

Requires ``control_mode = "attitude"``.
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
from scipy.spatial.transform import Rotation as R

import lsy_drone_racing.control.pp_racer_mpcc as base
import lsy_drone_racing.control.pp_racer_x as ppx
from lsy_drone_racing.control.pp_racer_mpcc import PPRacerMPCC
from lsy_drone_racing.control.pp_racer_x import G, PPRacerX

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
DCMD_MAX = 20.0  # [rad/s] roll/pitch command rate
DTHRUST_MAX = 20.0  # [N/s]
ATHETA_MAX = 30.0  # [m/s^2]
THRUST_FRAC = 0.95  # collective command bound (the rotor constraints do the rest)
ROTOR_FRAC = 0.97  # usable fraction of the per-rotor thrust range
V_THETA_MAX = 5.0
# Soft constraint penalties: [v_theta cap, speed cap, 8 rotor limits]
SLACK_V = (1e3, 1e4)  # L1 [1/(m/s)], L2 [1/(m/s)^2]
SLACK_ROTOR = (2e3, 2e5)  # L1 [1/N], L2 [1/N^2]
# Speed caps (loose profile + reveal caps)
CAP_V_MAX = 4.0
CAP_A_LAT = 11.0
CAP_A_MAX = 11.0
CAP_F_MAX = 15.0
CAP_V_UNSEEN = 2.42  # [m/s] within CAP_R_GATE of gates not revealed yet
CAP_R_GATE = 1.0
CAP_V_OBS = 2.5  # [m/s] within CAP_R_OBS of poles not revealed yet
CAP_R_OBS = 0.6
DIST_GAIN = 0.05  # disturbance observer gain (offset-free MPC)
QP_ITER_MAX = 50

_MODEL_VERSION = 2
_OCP_KEYS = (
    "N_HORIZON", "DT", "MU", "Q_CONTOUR", "Q_LAG", "Q_DRPY", "R_DCMD", "R_DTHRUST", "R_ATHETA",
    "R_YAW", "B_TUNNEL", "ALPHA", "MAX_TILT_CMD", "DCMD_MAX", "DTHRUST_MAX", "ATHETA_MAX",
    "THRUST_FRAC", "ROTOR_FRAC", "V_THETA_MAX", "SLACK_V", "SLACK_ROTOR", "QP_ITER_MAX",
)  # fmt: skip
_SOLVERS: dict[str, AcadosOcpSolver] = {}
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
        att=[base.ATT_A, base.ATT_B, base.ATT_C, base.THRUST_TAU, base.THRUST_GAIN],
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
    acc = z_body * base.THRUST_GAIN * thrust / m - cs.DM(kd) * vel - cs.DM([0, 0, G]) + d_acc
    ddrpy = cs.DM(base.ATT_A) * rpy + cs.DM(base.ATT_B) * drpy + cs.DM(base.ATT_C) * cmd_rpy
    f = cs.vertcat(vel, acc, drpy, ddrpy, (cmd_t - thrust) / base.THRUST_TAU, u[0:3], v_theta, u[3])

    # Rotor limits: torque needed for the predicted angular acceleration, mapped onto the rotors
    J = np.diag(np.asarray(dyn["J"], dtype=float))
    arm = float(dyn["L"])
    mix = np.asarray(dyn["mixing_matrix"], dtype=float)  # rows: roll, pitch, yaw torque signs
    delta = cs.vertcat(J[0] * ddrpy[0], J[1] * ddrpy[1]) / (4.0 * arm)  # [N] per rotor
    f_mean = base.THRUST_GAIN * thrust * t_scale / 4.0
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
    with open(out.parent / f"{model.name}.lock", "w") as lock:
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


class PPRacerMPCCPP(PPRacerMPCC):
    """MPCC++-style racing MPCC with rotor limits, command-rate inputs and a tunnel."""

    _THETA = 16  # progress variable in the state (see NX layout)

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the path and set up the solver.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        PPRacerX.__init__(self, obs, info, config)
        dyn = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        self._kd = -np.diag(self._drag) / self._m_nom
        self._solver = _build_solver(dyn, self._m_nom, self._kd, self._t_min, self._t_max)
        self._reset_mpc_state()

    def _reset_mpc_state(self):
        super()._reset_mpc_state()
        self._cmd = np.array([0.0, 0.0, self._m_nom * G])  # roll, pitch, thrust (nominal units)
        self._v_theta = 0.0

    @staticmethod
    def _dist_gain() -> float:
        return DIST_GAIN

    # -----------------------------------------------------------------------------------------
    # Path: tunnel widths and speed caps per sample
    # -----------------------------------------------------------------------------------------
    def _read_track(self, obs: dict[str, NDArray[np.floating]]):
        super()._read_track(obs)
        self._obst_seen = np.asarray(obs["obstacles_visited"], dtype=bool).copy()

    def _track_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        seen = np.asarray(obs["obstacles_visited"], dtype=bool)
        return super()._track_changed(obs) or bool(np.any(seen != self._obst_seen))

    def _build_plan(self):
        """Parent plan + tunnel half widths W(s), H(s) + loose speed cap v_cap(s)."""
        super()._build_plan()
        s, p = self._path_s, self._path_p
        d_pole = np.min(
            np.linalg.norm(p[:, None, :2] - self._obstacles_pos[None, :, :2], axis=2), axis=1
        )
        w = np.clip(d_pole - POLE_MARGIN, W_MIN, W_FREE)
        bump = self._path_bump
        self._path_w = w - (w - np.minimum(w, W_GATE)) * bump
        self._path_h = H_FREE - (H_FREE - W_GATE) * bump
        kappa = np.gradient(self._path_t, s, axis=0)
        kappa -= np.sum(kappa * self._path_t, axis=1)[:, None] * self._path_t
        v_cap = np.full(len(s), CAP_V_MAX)
        for gate, seen in zip(self._gates_pos, self._gates_seen):
            if not seen:
                near = np.linalg.norm(p - gate, axis=1) < CAP_R_GATE
                v_cap[near] = np.minimum(v_cap[near], CAP_V_UNSEEN)
        for obst, seen in zip(self._obstacles_pos, self._obst_seen):
            if not seen:
                near = np.linalg.norm(p[:, :2] - obst[:2], axis=1) < CAP_R_OBS
                v_cap[near] = np.minimum(v_cap[near], CAP_V_OBS)
        saved = (ppx.A_LAT_MAX, ppx.A_MAX, ppx.F_MAX)
        ppx.A_LAT_MAX, ppx.A_MAX, ppx.F_MAX = CAP_A_LAT, CAP_A_MAX, CAP_F_MAX
        try:
            self._path_vcap = np.maximum(self._speed_profile(s, self._path_t, kappa, v_cap), 0.3)
        finally:
            ppx.A_LAT_MAX, ppx.A_MAX, ppx.F_MAX = saved

    def _frame_at(self, theta: NDArray) -> tuple[NDArray, ...]:
        """Path point, tangent, normal (horizontal), binormal, W, H and v_cap at theta."""
        p, t, _, _ = self._path_at(theta)
        n = np.cross(np.array([0.0, 0.0, 1.0]), t)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(nn > 1e-3, n / np.maximum(nn, 1e-9), np.array([1.0, 0.0, 0.0]))
        b = np.cross(t, n)
        th = np.clip(theta, 0.0, self._path_s[-1])
        w = np.interp(th, self._path_s, self._path_w)
        h = np.interp(th, self._path_s, self._path_h)
        vc = np.interp(th, self._path_s, self._path_vcap)
        return p, t, n, b, w, h, vc

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

        if self._X is None:
            v0 = max(float(np.linalg.norm(obs["vel"])), base.V_THETA_MIN_GUESS)
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
        tilt = base.LOW_TILT if pos[2] < base.LOW_Z else MAX_TILT_CMD
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
            if status not in base._OK_STATUS:
                break
        self._n_iter = 1
        if status not in base._OK_STATUS:
            self._fails += 1
            self._X = self._U = None
            self._n_iter = base.ITERS_COLD
            solver.reset()
            return self._u_prev.copy()
        self._X = np.array([solver.get(k, "x") for k in range(n + 1)])
        self._U = np.array([solver.get(k, "u") for k in range(n)])
        self._v_pred = (self._X[1, 3:6].copy(), np.asarray(obs["vel"], dtype=float).copy())
        # Command for the coming interval: the command states at the end of the first stage
        self._cmd = self._X[1, 13:16].copy()
        return np.array([self._cmd[0], self._cmd[1], 0.0, self._cmd[2]])
