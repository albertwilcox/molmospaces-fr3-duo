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
from molmo_spaces.tasks.task_sampler_errors import ObjectPlacementError, RobotPlacementError
from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb
from molmo_spaces.utils.mujoco_scene_utils import place_object_near
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

    def _sample_place_robot_base_pose(self, env: CPUMujocoEnv) -> None:
        """Record a collision-free, receptacle-facing base pose as a *hint* for
        the place phase.

        This is only a starting candidate: the FSM performs a proper place-time
        base search (using the true held-object orientation) and will fall back
        to a ring of standoffs if this hint is not IK-feasible. We therefore keep
        the sampling cheap (collision-free placement only). On failure the field
        is left ``None`` and the FSM's ring search supplies candidates.
        """
        task_cfg = self.config.task_config
        sampler_cfg = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]
        receptacle = om.get_object_by_name(task_cfg.place_receptacle_name)
        robot_view = env.current_robot.robot_view

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
        if placed:
            task_cfg.place_robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
            log.info(
                f"[MOBILE PNP] Recorded place-hint base pose near "
                f"'{receptacle.name}' at {robot_view.base.pose[:2, 3]}."
            )
        else:
            task_cfg.place_robot_base_pose = None
            log.warning(
                f"[MOBILE PNP] Could not place robot near receptacle '{receptacle.name}'; "
                f"PLACE will rely on the FSM ring search."
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

    def _find_floor_geom_id(self, env: CPUMujocoEnv) -> int | None:
        """Return the geom id of the largest floor/room surface in the scene.

        Used to stand the place receptacle on the floor far from the pickup
        object (giving a genuine second navigation segment). Returns ``None`` if
        no floor/room body can be identified, in which case the caller falls
        back to same-surface placement.
        """
        model = env.current_model
        data = env.current_data
        om = env.object_managers[env.current_batch_index]

        best_geom: int | None = None
        best_area = -1.0
        for body_id in om.top_level_bodies():
            try:
                types = om.get_possible_object_types(body_id)
            except Exception:
                continue
            if not any(t in {"room", "floor", "Floor", "Room"} for t in types):
                continue
            adr = int(model.body_geomadr[body_id])
            num = int(model.body_geomnum[body_id])
            for gid in range(adr, adr + num):
                try:
                    _, dims = geom_aabb(model, data, [gid])
                except Exception:
                    continue
                area = float(dims[0] * dims[1])
                if area > best_area:
                    best_area = area
                    best_geom = gid
        return best_geom

    def _prepare_place_target(
        self,
        env: CPUMujocoEnv,
        place_target_name: str,
        pickup_obj_name: str,
        pickup_obj_pos: np.ndarray,
        supporting_geom_id: int,
    ) -> bool:
        """Stand the place receptacle(s) on the floor a real navigation distance
        from the pickup object, so mobile pick-and-place is genuinely
        navigate -> grasp -> navigate -> place.

        Falls back to the fixed-base same-surface placement when far-on-floor is
        disabled or no floor geom can be found.
        """
        sampler_cfg = self.config.task_sampler_config
        if not getattr(sampler_cfg, "far_place_on_floor", False):
            return super()._prepare_place_target(
                env, place_target_name, pickup_obj_name, pickup_obj_pos, supporting_geom_id
            )

        floor_geom_id = self._find_floor_geom_id(env)
        if floor_geom_id is None:
            log.warning(
                "[MOBILE PNP] No floor geom found; falling back to same-surface "
                "receptacle placement (short place-nav segment)."
            )
            return super()._prepare_place_target(
                env, place_target_name, pickup_obj_name, pickup_obj_pos, supporting_geom_id
            )

        om = env.object_managers[env.current_batch_index]
        for receptacle_name in self.active_receptacle_names:
            if not self._filter_place_target(env, pickup_obj_name, receptacle_name):
                log.info(f"Place receptacle {receptacle_name} fails filter size")
                if self.config.task_sampler_config.added_pickup_objects:
                    self._advance_to_next_added_pickupable(env)
                return False

            receptacle_id = om.get_object_body_id(receptacle_name)
            try:
                place_object_near(
                    data=env.current_data,
                    object_id=receptacle_id,
                    placement_point=pickup_obj_pos,
                    min_dist=sampler_cfg.far_min_object_to_receptacle_dist,
                    max_dist=sampler_cfg.far_max_object_to_receptacle_dist,
                    max_tries=sampler_cfg.max_place_receptacle_sampling_attempts,
                    supporting_geom_id=floor_geom_id,
                    z_eps=0.003,
                )
            except ObjectPlacementError:
                log.info(
                    f"[MOBILE PNP] Failed to stand receptacle {receptacle_name} on the "
                    f"floor far from the pickup object."
                )
                return False

        return True

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
