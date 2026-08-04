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
from typing import Any

import mujoco
import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.mobile_pick_and_place_task import MobilePickAndPlaceTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.task_sampler_errors import ObjectPlacementError, RobotPlacementError
from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom, place_object_near
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

    def _floor_top_z(self, env: CPUMujocoEnv) -> float:
        """Top-z of the scene floor (fallback 0.0), used as the reference height
        above which a candidate place surface counts as 'elevated'."""
        floor_geom_id = self._find_floor_geom_id(env)
        if floor_geom_id is None:
            return 0.0
        try:
            center, dims = geom_aabb(env.current_model, env.current_data, [floor_geom_id])
        except Exception:
            return 0.0
        return float(center[2] + dims[2] / 2.0)

    def _find_far_elevated_surface_geoms(
        self, env: CPUMujocoEnv, pickup_obj_pos: np.ndarray
    ) -> list[int]:
        """Discover elevated support surfaces (table / counter / shelf tops) a
        real navigation distance from the pickup object.

        Rather than guess which geoms are surfaces, we probe the surfaces that
        *already support scene objects*: for every top-level object we find the
        geom it rests on (``get_supporting_geom``) and keep the unique surfaces
        whose top is elevated above the floor, whose flat top is large enough for
        the receptacle, and whose horizontal distance from the pickup lies in the
        far band. Farther surfaces are preferred so the place-nav segment is
        genuinely exercised.
        """
        sampler_cfg = self.config.task_sampler_config
        model = env.current_model
        data = env.current_data
        om = env.object_managers[env.current_batch_index]

        floor_z = self._floor_top_z(env)
        min_top_z = floor_z + float(sampler_cfg.elevated_min_height_m)
        min_area = float(sampler_cfg.elevated_min_surface_area_m2)
        far_max = float(sampler_cfg.far_max_object_to_receptacle_dist)
        # When restricting the place to the pickup object's room, relax the lower
        # distance bound (intra-room distances are shorter) and resolve the
        # pickup object's room id for same-room filtering of candidate surfaces.
        same_room_only = bool(getattr(sampler_cfg, "same_room_place_only", False))
        if same_room_only:
            far_min = float(
                getattr(sampler_cfg, "same_room_min_object_to_receptacle_dist", 0.8)
            )
            pickup_room = self._room_id_of_name(self.config.task_config.pickup_obj_name)
        else:
            far_min = float(sampler_cfg.far_min_object_to_receptacle_dist)
            pickup_room = None
        pickup_xy = np.asarray(pickup_obj_pos)[:2]

        seen: set[int] = set()
        scored: list[tuple[float, int]] = []
        for body_id in om.top_level_bodies():
            name = om.get_object_name(body_id)
            if not name or om.is_excluded(name) or om.is_structural(name):
                continue
            try:
                geom_id = get_supporting_geom(data, int(body_id))
            except Exception:
                geom_id = None
            if geom_id is None or int(geom_id) < 1 or int(geom_id) in seen:
                continue
            seen.add(int(geom_id))
            try:
                center, dims = geom_aabb(model, data, [int(geom_id)])
            except Exception:
                continue
            top_z = float(center[2] + dims[2] / 2.0)
            area = float(dims[0] * dims[1])
            if top_z < min_top_z or area < min_area:
                continue
            dist = float(np.linalg.norm(np.asarray(center[:2]) - pickup_xy))
            if not (far_min <= dist <= far_max):
                continue
            # Same-room gate: the receptacle must stand on a surface in the same
            # room as the pickup object. The surface geom's supporting furniture
            # body carries the room id in its name (``..._<room_id>``).
            if pickup_room is not None:
                surf_room = self._surface_geom_room_id(model, om, int(geom_id))
                if surf_room is None or surf_room != pickup_room:
                    continue
            scored.append((dist, int(geom_id)))

        # Prefer farther surfaces (genuine place-nav segment) but keep all
        # candidates so placement can fall through to a nearer elevated surface.
        scored.sort(key=lambda t: -t[0])
        room_note = f" (same-room={pickup_room})" if pickup_room is not None else ""
        log.info(
            f"[MOBILE PNP] elevated-surface search: {len(scored)} candidate surface(s) "
            f"in [{far_min:.1f},{far_max:.1f}]m band above z={min_top_z:.2f}"
            f"{room_note} (probed {len(seen)} supporting geoms)."
        )
        return [gid for _, gid in scored]

    @staticmethod
    def _room_id_of_name(name: str | None) -> str | None:
        """Extract the room id from a scene object body name.

        Scene object bodies follow the ``{lemma}_{hash}_{count}_{body_idx}_{room_id}``
        convention (see ``molmo_spaces.housegen.utils.generate_body_name``), so
        the trailing underscore-delimited field is the room id. Returns ``None``
        for names that do not carry a numeric room suffix.
        """
        if not name:
            return None
        tail = name.rsplit("_", 1)[-1]
        return tail if tail.isdigit() else None

    def _surface_geom_room_id(
        self, model: Any, om: Any, geom_id: int
    ) -> str | None:
        """Room id of the furniture body that owns a candidate surface geom."""
        try:
            body_id = int(model.geom_bodyid[geom_id])
            root_body_id = int(model.body_rootid[body_id])
            name = om.get_object_name(root_body_id)
        except Exception:
            return None
        return self._room_id_of_name(name)

    def _prepare_place_target(
        self,
        env: CPUMujocoEnv,
        place_target_name: str,
        pickup_obj_name: str,
        pickup_obj_pos: np.ndarray,
        supporting_geom_id: int,
    ) -> bool:
        """Stand the place receptacle(s) on an *elevated* surface a real
        navigation distance from the pickup object, so mobile pick-and-place is
        genuinely navigate -> grasp -> navigate -> place and the place target is
        at a natural, reachable manipulation height (never on the floor).

        Order of preference:
        1. A far elevated surface (table/counter/shelf top) -- the desired case.
        2. (legacy, off by default) the floor far away, if ``far_place_on_floor``.
        3. The fixed-base same-surface placement (elevated, reachable, but a short
           place-nav segment) as a last resort so a valid episode is still
           produced instead of dropping the house.
        """
        sampler_cfg = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]

        if getattr(sampler_cfg, "far_place_on_elevated_surface", True):
            surface_geoms = self._find_far_elevated_surface_geoms(env, pickup_obj_pos)
            if surface_geoms and self._place_receptacles_on_surfaces(
                env, pickup_obj_name, surface_geoms
            ):
                return True
            log.info(
                "[MOBILE PNP] No far elevated surface worked for the receptacle; "
                "falling back."
            )

        if getattr(sampler_cfg, "far_place_on_floor", False):
            floor_geom_id = self._find_floor_geom_id(env)
            if floor_geom_id is not None and self._place_receptacles_on_floor(
                env, pickup_obj_name, pickup_obj_pos, floor_geom_id
            ):
                return True

        log.info(
            "[MOBILE PNP] Falling back to same-surface receptacle placement "
            "(short place-nav segment, still elevated)."
        )
        return super()._prepare_place_target(
            env, place_target_name, pickup_obj_name, pickup_obj_pos, supporting_geom_id
        )

    def _place_receptacles_on_surfaces(
        self, env: CPUMujocoEnv, pickup_obj_name: str, surface_geoms: list[int]
    ) -> bool:
        """Stand every active receptacle on the first far elevated surface that
        accommodates it. Returns False if no surface works for a receptacle."""
        sampler_cfg = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]
        model = env.current_model
        data = env.current_data

        for receptacle_name in self.active_receptacle_names:
            if not self._filter_place_target(env, pickup_obj_name, receptacle_name):
                log.info(f"Place receptacle {receptacle_name} fails filter size")
                if self.config.task_sampler_config.added_pickup_objects:
                    self._advance_to_next_added_pickupable(env)
                return False

            receptacle_id = om.get_object_body_id(receptacle_name)
            placed = False
            for geom_id in surface_geoms:
                try:
                    center, dims = geom_aabb(model, data, [int(geom_id)])
                except Exception:
                    continue
                reach = float(max(dims[0], dims[1]) / 2.0)
                try:
                    place_object_near(
                        data=data,
                        object_id=receptacle_id,
                        placement_point=np.asarray(center, dtype=float),
                        min_dist=0.0,
                        max_dist=reach,
                        max_tries=sampler_cfg.max_place_receptacle_sampling_attempts,
                        supporting_geom_id=int(geom_id),
                        z_eps=0.003,
                    )
                    placed = True
                    log.info(
                        f"[MOBILE PNP] Stood receptacle {receptacle_name} on far elevated "
                        f"surface geom {int(geom_id)} (top area~{dims[0]*dims[1]:.2f}m^2)."
                    )
                    break
                except ObjectPlacementError:
                    continue
            if not placed:
                log.info(
                    f"[MOBILE PNP] Could not stand receptacle {receptacle_name} on any "
                    f"of {len(surface_geoms)} far elevated surfaces."
                )
                return False

        return True

    def _place_receptacles_on_floor(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        pickup_obj_pos: np.ndarray,
        floor_geom_id: int,
    ) -> bool:
        """Legacy far-on-floor placement (kept behind ``far_place_on_floor``)."""
        sampler_cfg = self.config.task_sampler_config
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
