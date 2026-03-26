# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for environment action types."""

import numpy as np
import pytest

from agi.env.actions import (
    Action,
    ActionStatus,
    CartesianAction,
    CartesianPose,
    CartesianTwist,
    ControlMode,
    GripperAction,
    GripperCommand,
    JointAction,
    JointCommand,
    MobileBaseAction,
    BasePose,
    BaseVelocity,
    RobotAction,
    SkillAction,
)


class TestJointAction:
    """Tests for JointAction."""

    def test_from_arrays(self):
        names = ["j1", "j2", "j3"]
        values = np.array([0.1, 0.2, 0.3])
        action = JointAction.from_arrays(names, values)

        assert len(action.commands) == 3
        assert action.commands[0].name == "j1"
        np.testing.assert_almost_equal(action.commands[0].value, 0.1)

    def test_joint_names(self):
        names = ["shoulder", "elbow", "wrist"]
        values = np.array([1.0, 2.0, 3.0])
        action = JointAction.from_arrays(names, values)
        assert action.joint_names == ["shoulder", "elbow", "wrist"]

    def test_values_array(self):
        names = ["j1", "j2"]
        values = np.array([0.5, 1.5])
        action = JointAction.from_arrays(names, values)
        np.testing.assert_array_almost_equal(action.values, [0.5, 1.5])

    def test_mode_property(self):
        action = JointAction.from_arrays(
            ["j1"], np.array([0.0]), mode=ControlMode.VELOCITY
        )
        assert action.mode == ControlMode.VELOCITY

    def test_default_mode(self):
        action = JointAction()
        assert action.mode == ControlMode.POSITION


class TestCartesianAction:
    """Tests for CartesianAction."""

    def test_pose_command(self):
        pose = CartesianPose(
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        action = CartesianAction(pose=pose)
        assert action.is_pose_command
        assert not action.is_twist_command

    def test_twist_command(self):
        twist = CartesianTwist(
            linear=np.array([0.1, 0.0, 0.0]),
            angular=np.array([0.0, 0.0, 0.1]),
        )
        action = CartesianAction(twist=twist)
        assert action.is_twist_command
        assert not action.is_pose_command

    def test_from_matrix(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        pose = CartesianPose.from_matrix(matrix)
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])

    def test_zero_twist(self):
        twist = CartesianTwist.zero()
        np.testing.assert_array_equal(twist.linear, np.zeros(3))
        np.testing.assert_array_equal(twist.angular, np.zeros(3))


class TestGripperAction:
    """Tests for GripperAction."""

    def test_open(self):
        action = GripperAction.open()
        assert action.command == GripperCommand.OPEN
        assert action.position == 1.0

    def test_close(self):
        action = GripperAction.close()
        assert action.command == GripperCommand.CLOSE
        assert action.position == 0.0

    def test_grasp(self):
        action = GripperAction.grasp(force=0.5)
        assert action.command == GripperCommand.GRASP
        assert action.force == 0.5


class TestMobileBaseAction:
    """Tests for MobileBaseAction."""

    def test_move_forward(self):
        action = MobileBaseAction.move_forward(speed=1.0)
        assert action.velocity.linear_x == 1.0
        assert action.velocity.angular_z == 0.0

    def test_rotate(self):
        action = MobileBaseAction.rotate(angular_speed=0.5)
        assert action.velocity.angular_z == 0.5
        assert action.velocity.linear_x == 0.0

    def test_stop(self):
        action = MobileBaseAction.stop()
        assert action.velocity.linear_x == 0.0
        assert action.velocity.angular_z == 0.0


class TestRobotAction:
    """Tests for composite RobotAction."""

    def test_has_arm_command_joint(self):
        action = RobotAction(joint_action=JointAction())
        assert action.has_arm_command

    def test_has_arm_command_cartesian(self):
        action = RobotAction(cartesian_action=CartesianAction())
        assert action.has_arm_command

    def test_has_gripper_command(self):
        action = RobotAction(gripper_action=GripperAction.open())
        assert action.has_gripper_command

    def test_has_base_command(self):
        action = RobotAction(base_action=MobileBaseAction.stop())
        assert action.has_base_command

    def test_empty_robot_action(self):
        action = RobotAction()
        assert not action.has_arm_command
        assert not action.has_gripper_command
        assert not action.has_base_command


class TestSkillAction:
    """Tests for SkillAction."""

    def test_pick(self):
        action = SkillAction.pick("object_1")
        assert action.skill_name == "pick"
        assert action.target_object == "object_1"

    def test_place(self):
        location = np.array([1.0, 2.0, 0.5])
        action = SkillAction.place(location)
        assert action.skill_name == "place"
        np.testing.assert_array_equal(action.target_location, location)

    def test_navigate(self):
        location = np.array([5.0, 3.0, 0.0])
        action = SkillAction.navigate(location)
        assert action.skill_name == "navigate"
        np.testing.assert_array_equal(action.target_location, location)


class TestEnums:
    """Tests for action enums."""

    def test_control_modes(self):
        assert ControlMode.POSITION == "position"
        assert ControlMode.VELOCITY == "velocity"
        assert ControlMode.TORQUE == "torque"

    def test_action_status(self):
        assert ActionStatus.PENDING == "pending"
        assert ActionStatus.COMPLETED == "completed"
        assert ActionStatus.FAILED == "failed"
