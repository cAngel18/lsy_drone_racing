"""Control module for drone racing.

This module contains the base controller class that defines the interface for all controllers. Your
own controller goes in this module. It has to inherit from the base class and adhere to the same
function signatures.

To give you an idea of what you need to do, we also include some example implementations:

* [Controller][lsy_drone_racing.control.controller.Controller]: The abstract base class defining the
  interface for all controllers.
* [StateController][lsy_drone_racing.control.state_controller.StateController]:
  A controller that follows a pre-defined trajectory using cubic spline interpolation.
* [AttitudeController][lsy_drone_racing.control.attitude_controller.AttitudeController]: A
  controller that follows a pre-defined attitude using cubic spline interpolation.
* [MPPIController][lsy_drone_racing.control.mppi_controller.MPPIController]: A sampling-based
  Model Predictive Path Integral controller with online path planning for randomized tracks.
* [MPPINoPlanner][lsy_drone_racing.control.mppi_no_planner.MPPINoPlanner]: The same MPPI
  controller without a path planner, guided only by the gate geometry.
"""

from lsy_drone_racing.control.controller import Controller

__all__ = ["Controller"]
