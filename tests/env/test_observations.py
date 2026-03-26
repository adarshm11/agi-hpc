# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""
Unit tests for structured observation types.
Sprint 6 Phase 3.
"""

import numpy as np
import pytest
from agi.env.observations import (
    CameraObservation,
    LidarObservation,
    PointCloud,
    ImuObservation,
    JointState,
    JointStateObservation,
    Wrench,
    ForceTorqueObservation,
    ContactPoint,
    ContactObservation,
    Pose,
    Twist,
    OdometryObservation,
    RobotObservation,
)


class TestCameraObservation:
    """Tests for CameraObservation."""

    def test_basic_creation(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        cam = CameraObservation(timestamp=1.0, rgb=rgb)
        assert cam.height == 480
        assert cam.width == 640
        assert cam.has_depth is False
        assert cam.has_semantic is False

    def test_with_depth(self):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = np.ones((240, 320), dtype=np.float32)
        cam = CameraObservation(timestamp=1.0, rgb=rgb, depth=depth)
        assert cam.has_depth is True

    def test_project_to_3d(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        intrinsics = np.array([[500, 0, 50], [0, 500, 50], [0, 0, 1]], dtype=np.float64)
        cam = CameraObservation(
            timestamp=1.0, rgb=rgb, depth=depth, intrinsics=intrinsics
        )
        pt = cam.project_to_3d(50, 50)
        assert pt is not None
        np.testing.assert_allclose(pt[2], 2.0)

    def test_project_to_3d_no_depth(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        cam = CameraObservation(timestamp=1.0, rgb=rgb)
        assert cam.project_to_3d(10, 10) is None


class TestLidarObservation:
    """Tests for LidarObservation."""

    def test_basic_creation(self):
        points = np.random.rand(100, 3).astype(np.float32)
        pc = PointCloud(points=points)
        lidar = LidarObservation(timestamp=1.0, point_cloud=pc)
        assert lidar.num_points == 100

    def test_crop_box(self):
        points = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5]], dtype=np.float32)
        pc = PointCloud(points=points)
        cropped = pc.crop_box(np.array([0, 0, 0]), np.array([2, 2, 2]))
        assert cropped.num_points == 2


class TestImuObservation:
    """Tests for ImuObservation."""

    def test_basic_creation(self):
        imu = ImuObservation(timestamp=1.0)
        np.testing.assert_array_equal(imu.linear_acceleration, np.zeros(3))
        np.testing.assert_array_equal(imu.angular_velocity, np.zeros(3))

    def test_get_euler_with_orientation(self):
        # Identity quaternion [0, 0, 0, 1]
        imu = ImuObservation(timestamp=1.0, orientation=np.array([0.0, 0.0, 0.0, 1.0]))
        euler = imu.get_euler()
        assert euler is not None
        np.testing.assert_allclose(euler, np.zeros(3), atol=1e-10)

    def test_get_euler_without_orientation(self):
        imu = ImuObservation(timestamp=1.0)
        assert imu.get_euler() is None


class TestJointStateObservation:
    """Tests for JointStateObservation."""

    def test_basic_creation(self):
        joints = [
            JointState(name="j1", position=0.1, velocity=0.2),
            JointState(name="j2", position=0.3, velocity=0.4),
        ]
        obs = JointStateObservation(timestamp=1.0, joint_states=joints)
        np.testing.assert_array_equal(obs.positions, np.array([0.1, 0.3]))
        np.testing.assert_array_equal(obs.velocities, np.array([0.2, 0.4]))
        assert obs.joint_names == ["j1", "j2"]

    def test_get_joint(self):
        joints = [JointState(name="elbow", position=1.5, velocity=0.0)]
        obs = JointStateObservation(timestamp=1.0, joint_states=joints)
        j = obs.get_joint("elbow")
        assert j is not None
        assert j.position == 1.5
        assert obs.get_joint("missing") is None


class TestForceTorqueObservation:
    """Tests for ForceTorqueObservation."""

    def test_wrench_magnitude(self):
        w = Wrench(force=np.array([3.0, 4.0, 0.0]), torque=np.zeros(3))
        assert abs(w.magnitude - 5.0) < 1e-6

    def test_default_wrench(self):
        ft = ForceTorqueObservation(timestamp=1.0)
        assert ft.wrench.magnitude == 0.0


class TestContactObservation:
    """Tests for ContactObservation."""

    def test_no_contacts(self):
        obs = ContactObservation(timestamp=1.0)
        assert obs.num_contacts == 0
        assert obs.in_contact is False
        assert obs.total_force == 0.0

    def test_with_contacts(self):
        cp = ContactPoint(
            position=np.zeros(3), normal=np.array([0, 0, 1.0]), force=10.0
        )
        obs = ContactObservation(timestamp=1.0, contacts=[cp])
        assert obs.num_contacts == 1
        assert obs.in_contact is True
        assert obs.total_force == 10.0


class TestOdometryObservation:
    """Tests for OdometryObservation."""

    def test_identity_pose(self):
        pose = Pose.identity()
        np.testing.assert_array_equal(pose.position, np.zeros(3))
        np.testing.assert_array_equal(pose.orientation, np.array([0, 0, 0, 1.0]))

    def test_pose_to_matrix(self):
        pose = Pose.identity()
        mat = pose.to_matrix()
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat, np.eye(4), atol=1e-10)

    def test_odometry_default(self):
        odom = OdometryObservation(timestamp=1.0)
        assert odom.pose is not None
        assert odom.twist is not None


class TestRobotObservation:
    """Tests for RobotObservation."""

    def test_empty_robot_obs(self):
        obs = RobotObservation(timestamp=1.0)
        assert obs.get_primary_camera() is None
        assert obs.imu is None

    def test_with_camera(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        cam = CameraObservation(timestamp=1.0, rgb=rgb)
        obs = RobotObservation(timestamp=1.0, cameras={"primary": cam})
        assert obs.get_primary_camera() is cam

    def test_primary_camera_fallback(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        cam = CameraObservation(timestamp=1.0, rgb=rgb)
        obs = RobotObservation(timestamp=1.0, cameras={"front": cam})
        assert obs.get_primary_camera() is cam
