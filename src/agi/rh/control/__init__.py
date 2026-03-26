# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
RH Advanced Control Module.

Provides trajectory planning, real-time controllers, robot hardware
abstraction, and simulation wrappers for the Right Hemisphere
sensorimotor subsystem.
"""

try:
    from agi.rh.control.trajectory import (
        TrajectoryConfig,
        Waypoint,
        Trajectory,
        RRTPlanner,
        CHOMPPlanner,
        TrajectoryOptimizer,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from agi.rh.control.realtime import (
        PIDConfig,
        PIDController,
        MPCConfig,
        MPCController,
        ImpedanceConfig,
        ImpedanceController,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from agi.rh.control.robot_interface import (
        RobotInterface,
        ROS2BridgeConfig,
        ROS2Bridge,
        URDFLoader,
        HardwareAbstraction,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from agi.rh.control.simulation import (
        SimulationWrapper,
        MuJoCoWrapper,
        IsaacSimWrapper,
        UnityWrapper,
        GazeboWrapper,
        SimulationFactory,
    )
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "TrajectoryConfig",
    "Waypoint",
    "Trajectory",
    "RRTPlanner",
    "CHOMPPlanner",
    "TrajectoryOptimizer",
    "PIDConfig",
    "PIDController",
    "MPCConfig",
    "MPCController",
    "ImpedanceConfig",
    "ImpedanceController",
    "RobotInterface",
    "ROS2BridgeConfig",
    "ROS2Bridge",
    "URDFLoader",
    "HardwareAbstraction",
    "SimulationWrapper",
    "MuJoCoWrapper",
    "IsaacSimWrapper",
    "UnityWrapper",
    "GazeboWrapper",
    "SimulationFactory",
]
