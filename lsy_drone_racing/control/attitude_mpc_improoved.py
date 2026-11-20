"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the symbolic drone dynamics."""
    # For more info on the models, check out https://github.com/learnsyslab/crazyflow
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            1500.0,  # pos
            1500.0,  # pos
            1500.0,  # pos
            0.05,  # rpy
            0.05,  # rpy
            0.05,  # rpy
            5.0,  # vel
            5.0,  # vel
            5.0,  # vel
            0.02,  # drpy
            0.02,  # drpy
            0.02,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            0.5,  # rpy
            0.5,  # rpy
            0.5,  # rpy
            8.0,  # thrust
        ]
    )

    Q_e = 3.0 * Q  # stronger terminal cost (tuned)
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    #maximum tilt
    max_tilt_degree = 90
    max_tilt_rad = max_tilt_degree / 180 * 3.1416

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-max_tilt_rad, -max_tilt_rad, -max_tilt_rad])
    ocp.constraints.ubx = np.array([max_tilt_rad, max_tilt_rad, max_tilt_rad])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    total_thrust_min = parameters["thrust_min"] * 4
    total_thrust_max = parameters["thrust_max"] * 4
    ocp.constraints.lbu = np.array([-max_tilt_rad, -max_tilt_rad, -max_tilt_rad, total_thrust_min])
    ocp.constraints.ubu = np.array([max_tilt_rad, max_tilt_rad, max_tilt_rad, total_thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.globalization = "MERIT_BACKTRACKING" # performs line search
    ocp.solver_options.tol = 1e-4

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 15

    # set prediction horizon
    ocp.solver_options.tf = Tf

    cmake_builder = None
    # On Windows, use CMake with MinGW to build the generated MPC solver DLL.
    if os.name == "nt":
        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.build_dir = "build-mingw"

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
        cmake_builder=cmake_builder,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Waypoints are generated from the observed gate and obstacle positions (path planner from
        # pathplanner_cs.py) and regenerated whenever these observations change.
        self._t_total = 4.5  # s
        self._start_pos = np.array(obs["pos"])
        self._gates_pos = None
        self._obstacles_pos = None
        self._update_trajectory(obs["gates_pos"], obs["obstacles_pos"])

        self.drone_params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False
        self._max_tilt = 0.0  # Maximum tilt angle (rad) between body z-axis and world z-axis

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            print("returns true")
            self._finished = True

        # Replan if gate or obstacle positions changed (e.g. once they are within sensor range)
        self._update_trajectory(obs["gates_pos"], obs["obstacles_pos"])

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        tilt = np.arccos(np.clip(R.from_quat(obs["quat"]).as_matrix()[2, 2], -1.0, 1.0))
        self._max_tilt = max(self._max_tilt, float(tilt))
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Setting state reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]  # position
        # zero roll, pitch
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]  # yaw
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]  # velocity
        # zero drpy

        # Setting input reference (index > self._nx)
        # zero rpy
        # hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]  # position
        # zero roll, pitch
        yref_e[5] = self._waypoints_yaw[i + self._N]  # yaw
        yref_e[6:9] = self._waypoints_vel[i + self._N]  # velocity
        # zero drpy
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solving problem and getting first input
        status = self._acados_ocp_solver.solve()
        if status != 0:
            print(f"acados status {status} at tick {self._tick}")  # e.g. 2 = max iter, 4 = QP failure
            # if status == 2:
            #     self._acados_ocp_solver.print_statistics()
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished
    
    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        draw_points(sim, self._waypoints, rgba=(0.0, 1.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

    def episode_callback(self):
        """Reset the integral error."""
        print(f"Max tilt angle: {np.rad2deg(self._max_tilt):.2f} deg")
        self._max_tilt = 0.0
        self._tick = 0

    def _update_trajectory(
        self, gates_pos: NDArray[np.floating], obstacles_pos: NDArray[np.floating]
    ):
        """(Re)compute the reference trajectory if gate or obstacle positions have changed."""
        gates_pos, obstacles_pos = np.array(gates_pos), np.array(obstacles_pos)
        if (
            self._gates_pos is not None
            and np.array_equal(gates_pos, self._gates_pos)
            and np.array_equal(obstacles_pos, self._obstacles_pos)
        ):
            return
        self._gates_pos, self._obstacles_pos = gates_pos, obstacles_pos

        self._waypoints = self._get_waypoints(gates_pos, obstacles_pos)
        # Time allocation proportional to segment length (constant average speed) instead of equal
        # time per segment, which demands very high accelerations on long segments.
        seg_len = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        t = self._t_total * np.concatenate(([0.0], np.cumsum(seg_len))) / seg_len.sum()
        # Clamped: zero velocity at start and end. The default (not-a-knot) starts at ~5 m/s while
        # the drone is still on the ground, which makes the MPC saturate and crash.
        self._des_pos_spline = CubicSpline(t, self._waypoints, bc_type="clamped")
        self._des_vel_spline = self._des_pos_spline.derivative()
        t_samples = np.linspace(0, self._t_total, int(self._freq * self._t_total))
        self._waypoints_pos = self._des_pos_spline(t_samples)
        self._waypoints_vel = self._des_vel_spline(t_samples)
        self._waypoints_yaw = np.zeros(len(t_samples))

    def _get_support_points(self, gate_pos: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [
                [gate_pos[1][0] - 0.4, gate_pos[1][1] + 0.1, gate_pos[1][2] - 0.2],
                [gate_pos[0][0] - 0.3, gate_pos[0][1] + 0.3, gate_pos[0][2]],
                [gate_pos[2][0] + 0.9, gate_pos[2][1] + 0.3, gate_pos[2][2] - 0.3],
                [gate_pos[1][0] - 1.5, gate_pos[1][1] + 1.5, gate_pos[1][2]],
            ]
        )

    def _get_obstacle_avoid_pos(self, obstacles_pos: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [
                [obstacles_pos[0][0], obstacles_pos[0][1] - 0.3, 0.6],
                [obstacles_pos[1][0] + 0.25, obstacles_pos[1][1] - 0.3, 1],
                [obstacles_pos[2][0] - 0.2, obstacles_pos[2][1] + 0.35, 1.2],
                [obstacles_pos[3][0] + 0.1, obstacles_pos[3][1] + 0.3, 1.2],
                [obstacles_pos[1][0] + 0.25, obstacles_pos[1][1] - 0.3, 1.2],
            ]
        )

    def _get_waypoints(
        self, gate_pos: NDArray[np.floating], obstacle_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        gate_pos_copy = np.array(gate_pos)

        obstacle_avoid_pos = self._get_obstacle_avoid_pos(obstacle_pos)
        support_points = self._get_support_points(gate_pos)

        return np.array(
            [
                self._start_pos,
                obstacle_avoid_pos[0],
                gate_pos_copy[0],  # Gate 0
                obstacle_avoid_pos[1],
                gate_pos_copy[1],  # Gate 1
                support_points[0],
                support_points[1],
                support_points[2],
                gate_pos_copy[2],  # Gate 2
                obstacle_avoid_pos[2],
                obstacle_avoid_pos[3],
                gate_pos_copy[3],  # Gate 3
                obstacle_avoid_pos[4],
                gate_pos_copy[1],  # Gate 1
                support_points[3],
            ]
        )
