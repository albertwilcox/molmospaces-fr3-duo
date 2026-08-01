"""Task sampler for the combined mobile pick-and-place-with-navigation task.

Composition strategy:

* Scene construction (pickup candidates, place receptacles, grasp-collision
  bodies) is inherited unchanged from :class:`PickAndPlaceTaskSampler`.
* Pickup-object / place-receptacle selection is inherited via
  ``_configure_pick_and_place``. During that selection the robot is placed at a
  manip-feasible standoff near the pickup object (so grasp feasibility is checked
  from a base pose representative of where navigation will park).
* After selection, the mobile base is teleported to a far *navigable* start pose
  so the episode genuinely exercises navigation before manipulation.
"""

import logging

import mujoco
import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.mobile_pick_and_place_task import MobilePickAndPlaceTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.task_sampler_errors import RobotPlacementError
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


class MobilePickAndPlaceTaskSampler(PickAndPlaceTaskSampler):
    """Sampler for :class:`MobilePickAndPlaceTask`."""

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Place the mobile robot at a manip-feasible standoff near the pickup
        object (floor height), so the inherited grasp-feasibility check runs from
        a pose representative of the eventual parked base.

        Overrides the fixed-base version, which drops the robot to a tabletop
        ``robot_object_z_offset`` height; a mobile base must stay on the floor.
        """
        task_cfg = self.config.task_config
        sampler_cfg = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()

        robot_view = env.current_robot.robot_view
        robot_placed = env.place_robot_near(
            robot_view=robot_view,
            target=pickup_obj,
            max_tries=sampler_cfg.max_robot_placement_attempts,
            sampling_radius_range=sampler_cfg.manip_standoff_radius_range,
            robot_safety_radius=sampler_cfg.robot_safety_radius,
            preserve_z=sampler_cfg.mobile_base_z,
            face_target=True,
            check_camera_visibility=False,
            excluded_positions=self.used_robot_positions[pickup_obj.name],
            save_visibility_frames_dir=self.config.output_dir,
        )
        if not robot_placed:
            raise RobotPlacementError(
                f"Failed to place mobile robot near pickup object: {pickup_obj.name}"
            )

        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

    def _place_pose_reachable(self, env: CPUMujocoEnv, receptacle) -> bool:
        """Return True if the arm can IK-reach a placement point over the
        receptacle top from the robot's *current* base pose.

        Mirrors the pickup grasp-feasibility gate for the place side: a base
        pose that is collision-free but leaves the receptacle past the arm's
        reach is useless, so we require a reachable place point before recording
        the pose. The placement point is the receptacle-top centre; a small yaw
        sweep of a tool-down orientation is tried (the place primitive has yaw
        freedom), and both grippers are considered (bimanual).
        """
        robot = env.current_robot
        kin = robot.kinematics
        robot_view = robot.robot_view
        base_pose = robot_view.base.pose
        q0 = robot_view.get_qpos_dict()

        center, size = body_aabb(env.current_data.model, env.current_data, receptacle.object_id)
        top_z = center[2] + size[2] / 2
        # Small clearance above the top surface, matching the place primitive.
        target_xyz = np.array([center[0], center[1], top_z + 0.05])

        for gripper_mg_id in robot_view.get_gripper_movegroup_ids():
            for yaw in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
                c, s = np.cos(yaw), np.sin(yaw)
                # Tool z-axis pointing down; x rotated by yaw about vertical.
                rot = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])
                target = np.eye(4)
                target[:3, :3] = rot
                target[:3, 3] = target_xyz
                jp = kin.ik(gripper_mg_id, target, None, q0, base_pose=base_pose)
                if jp is not None:
                    return True
        return False

    def _sample_place_robot_base_pose(self, env: CPUMujocoEnv) -> None:
        """Record a place-feasible base pose near the receptacle.

        Mirrors :meth:`_sample_and_place_robot` for the receptacle: place the
        mobile base at a collision-free, receptacle-facing standoff (floor
        height) and record it in ``task_config.place_robot_base_pose`` so the FSM
        can navigate the base there for the PLACE phase, instead of parking at the
        closest navigable cell (which leaves the receptacle past the arm's reach).

        Each candidate base pose is additionally gated on place-IK reachability
        (:meth:`_place_pose_reachable`) so the recorded pose can actually reach a
        placement point on the receptacle top. On failure the field is left
        ``None`` and the FSM falls back to sampling.
        """
        task_cfg = self.config.task_config
        sampler_cfg = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]
        receptacle = om.get_object_by_name(task_cfg.place_receptacle_name)
        robot_view = env.current_robot.robot_view

        max_outer = sampler_cfg.max_robot_placement_attempts
        for _ in range(max_outer):
            placed = env.place_robot_near(
                robot_view=robot_view,
                target=receptacle,
                max_tries=sampler_cfg.max_robot_placement_attempts,
                sampling_radius_range=sampler_cfg.manip_standoff_radius_range,
                robot_safety_radius=sampler_cfg.robot_safety_radius,
                preserve_z=sampler_cfg.mobile_base_z,
                face_target=True,
                check_camera_visibility=False,
            )
            if not placed:
                break
            mujoco.mj_forward(env.current_model, env.current_data)
            if self._place_pose_reachable(env, receptacle):
                task_cfg.place_robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
                log.info(
                    f"[MOBILE PNP] Recorded place-feasible (IK-reachable) base pose near "
                    f"'{receptacle.name}' at {robot_view.base.pose[:2, 3]}."
                )
                return

        task_cfg.place_robot_base_pose = None
        log.warning(
            f"[MOBILE PNP] Could not place robot at an IK-reachable pose near receptacle "
            f"'{receptacle.name}'; PLACE nav will fall back to goal sampling."
        )

    def _place_robot_at_nav_start(self, env: CPUMujocoEnv) -> None:
        """Move the mobile base to a far, navigable start pose facing anywhere.

        On failure (no navigable free cell in the requested radius band) the
        robot is left at the manip-feasible pose near the object, yielding a
        trivially-short navigation phase rather than an aborted sample.
        """
        sampler_cfg = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        robot_view = env.current_robot.robot_view

        placed = env.place_robot_near(
            robot_view=robot_view,
            target=pickup_obj,
            max_tries=sampler_cfg.max_robot_placement_attempts,
            sampling_radius_range=sampler_cfg.nav_start_radius_range,
            robot_safety_radius=sampler_cfg.robot_safety_radius,
            preserve_z=sampler_cfg.mobile_base_z,
            face_target=False,
            check_camera_visibility=False,
        )
        if not placed:
            log.warning(
                "[MOBILE PNP] Could not place robot at a far nav-start pose; "
                "keeping the manip-feasible pose near the object."
            )
        else:
            log.info(
                f"[MOBILE PNP] Robot placed at nav-start pose "
                f"{robot_view.base.pose[:2, 3]} (target={pickup_obj.name})."
            )

    def _sample_task(self, env: CPUMujocoEnv) -> MobilePickAndPlaceTask:
        # Select pickup object + place receptacle and populate the task config
        # (this also places the robot near the object for grasp feasibility).
        self._configure_pick_and_place(env)

        # Record a place-feasible base pose near the receptacle (while the scene
        # is settled) so the FSM can navigate there for the PLACE phase.
        self._sample_place_robot_base_pose(env)

        # Now relocate the base to a navigable start pose far from the object.
        self._place_robot_at_nav_start(env)

        mujoco.mj_forward(env.current_model, env.current_data)
        return MobilePickAndPlaceTask(env, self.config)
