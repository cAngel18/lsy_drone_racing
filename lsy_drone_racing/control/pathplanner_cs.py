"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class StateController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Same waypoints as in the attitude controller. Determined by trial and error.
        self._t_total = 9 # s
        self._start_pos = obs["pos"]
            
        gate_pos = obs["gates_pos"]
        obstacle_pos = obs["obstacles_pos"]
        self._waypoints = self._get_waypoints(gate_pos, obstacle_pos)
        t = np.linspace(0, self._t_total, len(self._waypoints))
        self._des_pos_spline = CubicSpline(t, self._waypoints)

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz] as a
            numpy array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True
          
        gate_pos = obs["gates_pos"]    
        obstacle_pos = obs["obstacles_pos"]
        self._waypoints = self._get_waypoints(gate_pos, obstacle_pos)
        t_vec = np.linspace(0, self._t_total, len(self._waypoints))
        self._des_pos_spline = CubicSpline(t_vec, self._waypoints)

        des_pos = self._des_pos_spline(t)
        des_yaw = 0.0
        des_quat = R.from_euler("z", des_yaw).as_quat()
        action = np.concatenate((des_pos, np.zeros(6), des_quat, np.zeros(3)), dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        draw_points(sim, self._waypoints, rgba=(0.0, 1.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
    
    
    def _get_support_points(self, gate_pos: NDArray[np.floating], obstacle_pos: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [
                [gate_pos[1][0]-0.4, gate_pos[1][1]+0.1, gate_pos[1][2]-0.2],
                [gate_pos[0][0]-0.3, gate_pos[0][1]+0.3, gate_pos[0][2]],
                [gate_pos[2][0]+0.9, gate_pos[2][1]+0.3, gate_pos[2][2]-0.3],
                [gate_pos[1][0]-0.5, gate_pos[1][1]+0.5, gate_pos[1][2]],
            ]
        )
    
    def _get_obstacle_avoid_pos(self, obstacles_pos: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [
                [obstacles_pos[0][0], obstacles_pos[0][1]-0.3, 0.6],
                [obstacles_pos[1][0]+0.25, obstacles_pos[1][1]-0.3, 1],
                [obstacles_pos[2][0]-0.2, obstacles_pos[2][1]+0.35, 1.1],
                [obstacles_pos[3][0]+0.1 , obstacles_pos[3][1]+0.3, 1.15],
                [obstacles_pos[1][0]+0.25, obstacles_pos[1][1]-0.3, 1.2],
            ]
        )
    
    def _get_waypoints(self, gate_pos: NDArray[np.floating], obstacle_pos: NDArray[np.floating]) -> NDArray[np.floating]:
        gate_pos_copy = np.array(gate_pos)
        gate_pos_copy[:, 2] -= 0.2
        
        obstacle_avoid_pos = self._get_obstacle_avoid_pos(obstacle_pos)
        support_points = self._get_support_points(gate_pos, obstacle_pos)
        
        waypoints = np.array(
            [
                self._start_pos,
                obstacle_avoid_pos[0],
                gate_pos_copy[0],   # Gate 0 [0.5, 0.25, 0.7]
                obstacle_avoid_pos[1],
                gate_pos_copy[1],  # Gate 1 [1.05, 0.75, 1.2]
                support_points[0],
                support_points[1],
                support_points[2],
                gate_pos_copy[2],  # Gate 2 [-1.0, -0.25, 0.7]
                obstacle_avoid_pos[2],
                obstacle_avoid_pos[3],
                gate_pos_copy[3],  # Gate 3 [0.0, -0.75, 1.2]
                obstacle_avoid_pos[4],
                gate_pos_copy[1],  # Gate 1 [1.05, 0.75, 1.2]
                support_points[3],
            ]
        )
        return waypoints
        