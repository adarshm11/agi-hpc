# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.rh.control.realtime module."""

import pytest
import numpy as np

from agi.rh.control.realtime import (
    PIDConfig,
    PIDController,
    MPCConfig,
    MPCController,
    ImpedanceConfig,
    ImpedanceController,
)


class TestPIDController:
    """Tests for the PIDController class."""

    def test_init_defaults(self):
        """PIDController initialises with default PIDConfig values."""
        pid = PIDController()
        assert pid._config.kp == 1.0
        assert pid._config.ki == 0.0
        assert pid._config.kd == 0.0
        assert pid._integral == 0.0
        assert pid._prev_error is None

    def test_init_custom_config(self):
        """PIDController respects a custom PIDConfig."""
        cfg = PIDConfig(kp=5.0, ki=0.2, kd=1.0, output_min=-10.0, output_max=10.0)
        pid = PIDController(config=cfg)
        assert pid._config.kp == 5.0
        assert pid._config.ki == 0.2
        assert pid._config.kd == 1.0
        assert pid._config.output_min == -10.0
        assert pid._config.output_max == 10.0

    def test_proportional_only(self):
        """Pure P-control returns kp * error."""
        cfg = PIDConfig(kp=2.0, ki=0.0, kd=0.0)
        pid = PIDController(config=cfg)
        output = pid.update(setpoint=10.0, measured=7.0, dt=0.01)
        assert output == pytest.approx(6.0)

    def test_output_clamping(self):
        """Output is clamped between output_min and output_max."""
        cfg = PIDConfig(kp=10.0, ki=0.0, kd=0.0, output_min=-5.0, output_max=5.0)
        pid = PIDController(config=cfg)
        output = pid.update(setpoint=100.0, measured=0.0, dt=0.01)
        assert output == pytest.approx(5.0)

    def test_reset_clears_state(self):
        """reset() zeroes integral and clears previous error."""
        pid = PIDController(config=PIDConfig(kp=1.0, ki=1.0))
        pid.update(setpoint=10.0, measured=0.0, dt=1.0)
        pid.reset()
        assert pid._integral == 0.0
        assert pid._prev_error is None
        assert pid._last_time is None

    def test_update_vector(self):
        """update_vector computes PID output for multi-dimensional arrays."""
        cfg = PIDConfig(kp=1.0, ki=0.0, kd=0.0)
        pid = PIDController(config=cfg)
        sp = np.array([1.0, 2.0, 3.0])
        meas = np.array([0.0, 0.0, 0.0])
        result = pid.update_vector(sp, meas, dt=0.01)
        np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))


class TestMPCController:
    """Tests for the MPCController class."""

    def test_init_defaults(self):
        """MPCController initialises with default MPCConfig values."""
        mpc = MPCController()
        assert mpc._config.horizon == 20
        assert mpc._config.state_dim == 6
        assert mpc._config.control_dim == 3

    def test_init_custom_config(self):
        """MPCController respects a custom MPCConfig."""
        cfg = MPCConfig(horizon=50, control_dim=4, q_weight=2.0)
        mpc = MPCController(config=cfg)
        assert mpc._config.horizon == 50
        assert mpc._config.control_dim == 4
        assert mpc._config.q_weight == 2.0

    def test_compute_returns_ndarray(self):
        """compute() returns an ndarray with length equal to control_dim."""
        cfg = MPCConfig(state_dim=6, control_dim=3)
        mpc = MPCController(config=cfg)
        state = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        reference = np.array([2.0, 3.0, 4.0, 0.0, 0.0, 0.0])
        action = mpc.compute(state, reference)
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)

    def test_compute_proportional_response(self):
        """Stub compute is proportional: q_weight * (ref - state) for control dims."""
        cfg = MPCConfig(control_dim=3, q_weight=1.0)
        mpc = MPCController(config=cfg)
        state = np.zeros(6)
        reference = np.array([3.0, 4.0, 5.0, 0.0, 0.0, 0.0])
        action = mpc.compute(state, reference)
        np.testing.assert_allclose(action, np.array([3.0, 4.0, 5.0]))

    def test_compute_clamped(self):
        """Actions are clipped to [control_min, control_max]."""
        cfg = MPCConfig(
            control_dim=2, control_min=-1.0, control_max=1.0, q_weight=100.0
        )
        mpc = MPCController(config=cfg)
        state = np.zeros(6)
        reference = np.array([5.0, -5.0, 0.0, 0.0, 0.0, 0.0])
        action = mpc.compute(state, reference)
        np.testing.assert_allclose(action, np.array([1.0, -1.0]))

    def test_reset_is_callable(self):
        """reset() can be called without error."""
        mpc = MPCController()
        mpc.reset()  # no-op but should not raise


class TestImpedanceController:
    """Tests for the ImpedanceController class."""

    def test_init_defaults(self):
        """ImpedanceController initialises with default ImpedanceConfig values."""
        ic = ImpedanceController()
        assert ic._config.stiffness == 100.0
        assert ic._config.damping == 20.0
        assert ic._config.dim == 3
        assert ic._stiffness_matrix.shape == (3, 3)

    def test_compute_zero_error(self):
        """Force is zero when position and velocity errors are both zero."""
        ic = ImpedanceController()
        z = np.zeros(3)
        force = ic.compute(desired_pos=z, desired_vel=z, current_pos=z, current_vel=z)
        np.testing.assert_allclose(force, np.zeros(3))

    def test_compute_position_error(self):
        """Force from stiffness when only position error is present."""
        cfg = ImpedanceConfig(stiffness=50.0, damping=0.0, dim=3, force_limit=1000.0)
        ic = ImpedanceController(config=cfg)
        desired = np.array([1.0, 0.0, 0.0])
        z = np.zeros(3)
        force = ic.compute(
            desired_pos=desired, desired_vel=z, current_pos=z, current_vel=z
        )
        np.testing.assert_allclose(force, np.array([50.0, 0.0, 0.0]))

    def test_force_limit_clamping(self):
        """Force magnitude is clamped to force_limit."""
        cfg = ImpedanceConfig(stiffness=1000.0, damping=0.0, dim=3, force_limit=10.0)
        ic = ImpedanceController(config=cfg)
        desired = np.array([10.0, 0.0, 0.0])
        z = np.zeros(3)
        force = ic.compute(
            desired_pos=desired, desired_vel=z, current_pos=z, current_vel=z
        )
        assert np.linalg.norm(force) == pytest.approx(10.0, abs=1e-9)

    def test_set_stiffness(self):
        """set_stiffness() updates the stiffness matrix diagonals."""
        ic = ImpedanceController()
        ic.set_stiffness(300.0)
        assert ic._config.stiffness == 300.0
        np.testing.assert_allclose(np.diag(ic._stiffness_matrix), [300.0, 300.0, 300.0])

    def test_set_damping(self):
        """set_damping() updates the damping matrix diagonals."""
        ic = ImpedanceController()
        ic.set_damping(50.0)
        assert ic._config.damping == 50.0
        np.testing.assert_allclose(np.diag(ic._damping_matrix), [50.0, 50.0, 50.0])
