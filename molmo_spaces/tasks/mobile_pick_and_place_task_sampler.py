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
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, descendant_geoms, geom_aabb
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom, place_object_near
from molmo_spaces.utils.pose import pose_mat_to_7d, pos_quat_to_pose_mat
from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask, load_grasps_for_object

log = logging.getLogger(__name__)

# Move-group id of the mobile base (matches the FSM policy's ``_BASE_MG_ID``).
# Excluded from the sample-time place-reachability IK probe so the base stays
# pinned at the probed standoff instead of sliding to reach the target.
_BASE_MG_ID_SAMPLER = "base"


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

        # Gate: the collision-only grasp check (inherited from PickTaskSampler)
        # accepts an object as long as >=1 grasp is non-colliding, but says
        # nothing about whether the ARM can actually reach any grasp with the
        # base frozen at this standoff. That optimism is the dominant runtime
        # failure ("PICK build failed: no reachable base standoff" ->
        # no_verified_grasp): small/flat objects (e.g. remotes on a surface)
        # whose only grasps sit outside the arm's base-locked reach envelope.
        # Re-run the SAME base-locked IK the FSM uses at runtime and reject the
        # standoff if no grasp's pregrasp+grasp pair is reachable, so a different
        # (graspable) pickup object is sampled instead.
        if getattr(sampler_cfg, "manip_reach_gate_enabled", True):
            if not self._pick_reachable_base_locked(env, pickup_obj, robot_view):
                raise RobotPlacementError(
                    "No base-locked-IK-reachable grasp for pickup object "
                    f"{pickup_obj.name} at the sampled standoff."
                )

        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

    def _arm_move_group_ids_for(self, robot_view) -> list[str]:
        """Arm move groups (arm only; never the holonomic base or the gripper).

        Mirrors ``_MobileManipPlannerPolicy._arm_move_group_ids`` so the sample-
        time reachability gate solves IK over exactly the joints the runtime
        unlocks during a base-frozen grasp.
        """
        gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
        return [
            mg
            for mg in robot_view.move_group_ids()
            if mg not in gripper_mgs and mg != _BASE_MG_ID_SAMPLER
        ]

    def _pick_reachable_base_locked(self, env: CPUMujocoEnv, pickup_obj, robot_view) -> bool:
        """Whether at least one non-colliding grasp of ``pickup_obj`` has BOTH
        its grasp pose and its pregrasp stand-off base-locked IK-reachable from
        the robot's current (parked) base pose.

        This is the sample-time analogue of the FSM's runtime standoff search:
        it uses the identical ``kinematics.ik`` call with the base move group
        locked, so an object that passes here is one the arm can genuinely reach
        without the base moving -- eliminating the ``no reachable base standoff``
        -> ``no_verified_grasp`` failures that dominate mobile pick.
        """
        asset_uid = self.get_asset_uid_from_object(env, pickup_obj.name)
        if not asset_uid:
            return True  # no asset metadata -> can't gate; defer to runtime.
        try:
            _gripper, cached_grasps = load_grasps_for_object(asset_uid, 512)
        except (ValueError, KeyError):
            return True  # no grasp file -> handled elsewhere; don't reject here.
        if len(cached_grasps) == 0:
            return True

        object_pose = pos_quat_to_pose_mat(pickup_obj.position, pickup_obj.quat)
        grasp_poses_world = object_pose @ cached_grasps
        try:
            noncolliding = get_noncolliding_grasp_mask(
                env.current_model, env.current_data, grasp_poses_world, 64
            )
        except (KeyError, ValueError):
            noncolliding = np.ones(len(grasp_poses_world), dtype=bool)
        feasible_world = grasp_poses_world[noncolliding]
        if len(feasible_world) == 0:
            return False

        kinematics = env.current_robot.kinematics
        base_pose = robot_view.base.pose
        arm_mgs = self._arm_move_group_ids_for(robot_view)
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        q0 = robot_view.get_qpos_dict()
        pregrasp_z = self.config.policy_config.manip_policy_config.pregrasp_z_offset

        # Cap the number of IK solves so a pathological object can't blow up
        # sample time; grasps are already cost/collision-filtered upstream.
        max_checks = int(getattr(self.config.task_sampler_config, "manip_reach_gate_max_grasps", 48))
        for grasp_world in feasible_world[:max_checks]:
            if kinematics.ik(gripper_mg_id, grasp_world, arm_mgs, q0, base_pose) is None:
                continue
            pregrasp_world = grasp_world.copy()
            pregrasp_world[:3, 3] -= pregrasp_z * pregrasp_world[:3, 2]
            if kinematics.ik(gripper_mg_id, pregrasp_world, arm_mgs, q0, base_pose) is None:
                continue
            return True
        return False

    def _sample_place_robot_base_pose(self, env: CPUMujocoEnv) -> None:
        """Record the place-phase base nav goal near the receptacle.

        Prefers the IK-verified standoff found by the sample-time place-
        reachability probe (:meth:`_find_place_standoff`), so navigation drives
        the base straight to a pose from which the PLACE phase builds in place
        (no ring search, no teleport). Falls back to a cheap collision-free,
        receptacle-facing placement when no verified standoff is available (probe
        disabled or fail-open); the FSM's ring search then supplies candidates.
        """
        task_cfg = self.config.task_config
        sampler_cfg = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]
        receptacle = om.get_object_by_name(task_cfg.place_receptacle_name)
        robot_view = env.current_robot.robot_view

        verified = getattr(self, "_verified_place_base_pose", None)
        if verified is not None:
            task_cfg.place_robot_base_pose = pose_mat_to_7d(np.asarray(verified)).tolist()
            log.info(
                f"[MOBILE PNP] Recorded IK-verified place standoff near "
                f"'{receptacle.name}' at {np.asarray(verified)[:2, 3]}."
            )
            return

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
        exclude_enclosed = bool(
            getattr(sampler_cfg, "place_exclude_enclosed_container_surfaces", False)
        )
        n_enclosed_skipped = 0
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
            # Enclosed-container gate: skip surfaces owned by an appliance the
            # base cannot maneuver into to place (fridge shelf, oven rack, etc.).
            # Only prunes candidates; open surfaces remain, so a house is never
            # exhausted by this filter.
            if exclude_enclosed and self._surface_geom_is_enclosed_container(
                model, om, int(geom_id)
            ):
                n_enclosed_skipped += 1
                continue
            scored.append((dist, int(geom_id)))

        # Prefer farther surfaces (genuine place-nav segment) but keep all
        # candidates so placement can fall through to a nearer elevated surface.
        scored.sort(key=lambda t: -t[0])
        room_note = f" (same-room={pickup_room})" if pickup_room is not None else ""
        enclosed_note = (
            f", {n_enclosed_skipped} enclosed-container surface(s) skipped"
            if exclude_enclosed and n_enclosed_skipped
            else ""
        )
        log.info(
            f"[MOBILE PNP] elevated-surface search: {len(scored)} candidate surface(s) "
            f"in [{far_min:.1f},{far_max:.1f}]m band above z={min_top_z:.2f}"
            f"{room_note} (probed {len(seen)} supporting geoms){enclosed_note}."
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

    def _find_furniture_top_geoms(
        self, env: CPUMujocoEnv, pickup_obj_pos: np.ndarray
    ) -> list[int]:
        """Discover the *top* geom of every static furniture body in the far
        band, WITHOUT requiring the furniture to already support a scene object.

        ``_find_far_elevated_surface_geoms`` finds surfaces by probing what each
        scene object rests on, so a bare bed/table (nothing on it) is invisible
        to it -- the main reason furniture placement previously "made little
        progress". This finder instead enumerates furniture bodies directly and
        returns, per body, the descendant geom whose AABB top is highest and flat
        enough to drop an object on. Used only in broad furniture-place mode.

        Eligibility per body: top-level, non-structural, non-excluded, STATIC (no
        free joint -> not a pickup object), NOT an enclosed articulable container
        (fridge/cabinet ... -- unreachable interiors), with a top geom above the
        elevated-height floor, of at least the broad min half-extent, in the far
        distance band, and (when restricting) in the pickup object's room."""
        sampler_cfg = self.config.task_sampler_config
        model = env.current_model
        data = env.current_data
        om = env.object_managers[env.current_batch_index]

        floor_z = self._floor_top_z(env)
        min_top_z = floor_z + float(sampler_cfg.elevated_min_height_m)
        min_half = float(getattr(sampler_cfg, "broad_furniture_min_half_extent_m", 0.20))
        far_max = float(sampler_cfg.far_max_object_to_receptacle_dist)
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
        pickup_root = int(
            model.body_rootid[int(om.get_object_body_id(self.config.task_config.pickup_obj_name))]
        )

        scored: list[tuple[float, int]] = []
        for body_id in om.top_level_bodies():
            root_id = int(model.body_rootid[int(body_id)])
            if root_id == pickup_root:
                continue
            name = om.get_object_name(int(body_id))
            if not name or om.is_excluded(name) or om.is_structural(name):
                continue
            # Static only: a body with a free joint is a movable object, not
            # furniture to place onto.
            try:
                if om.has_free_joint(name):
                    continue
            except Exception:
                pass
            # Never an enclosed container interior (fridge/cabinet ...).
            if self._surface_geom_is_enclosed_container_body(model, om, int(body_id)):
                continue
            # Furniture-type filter: skip poor drop targets (chairs/stools ...)
            # and, when allowlist-only, require a known flat-surface type. This
            # avoids wasting the (slow) standoff probe on furniture the base can
            # never stand off from -- the dominant cause of the low realized
            # furniture-placement rate.
            lname = name.lower()
            deny = tuple(getattr(sampler_cfg, "broad_furniture_deny_substrings", ()))
            if any(sub in lname for sub in deny):
                continue
            if bool(getattr(sampler_cfg, "broad_furniture_type_allowlist_only", False)):
                allow = tuple(getattr(sampler_cfg, "broad_furniture_allow_substrings", ()))
                if allow and not any(sub in lname for sub in allow):
                    continue
            # Highest flat descendant geom = the top surface.
            try:
                geom_ids = descendant_geoms(model, int(body_id), True)
            except Exception:
                geom_ids = []
            best_geom = None
            best_top_z = -np.inf
            for gid in geom_ids:
                try:
                    center, dims = geom_aabb(model, data, [int(gid)])
                except Exception:
                    continue
                if float(min(dims[0], dims[1]) / 2.0) < min_half:
                    continue  # thin/narrow geom -- not a drop surface.
                top_z = float(center[2] + dims[2] / 2.0)
                if top_z > best_top_z:
                    best_top_z = top_z
                    best_geom = int(gid)
            if best_geom is None or best_top_z < min_top_z:
                continue
            try:
                bcenter, _bdims = geom_aabb(model, data, [best_geom])
            except Exception:
                continue
            dist = float(np.linalg.norm(np.asarray(bcenter[:2]) - pickup_xy))
            if not (far_min <= dist <= far_max):
                continue
            if pickup_room is not None:
                surf_room = self._surface_geom_room_id(model, om, best_geom)
                if surf_room is None or surf_room != pickup_room:
                    continue
            scored.append((dist, best_geom))

        scored.sort(key=lambda t: -t[0])  # farther first (genuine place-nav).
        log.info(
            f"[MOBILE PNP] furniture-top search: {len(scored)} static furniture "
            f"surface(s) in [{far_min:.1f},{far_max:.1f}]m band above z={min_top_z:.2f}."
        )
        return [gid for _, gid in scored]

    def _surface_geom_is_enclosed_container_body(
        self, model: Any, om: Any, body_id: int
    ) -> bool:
        """Enclosed-container test keyed on a furniture BODY id (see
        ``_surface_geom_is_enclosed_container`` which keys on a geom id)."""
        try:
            name = om.get_object_name(int(model.body_rootid[int(body_id)]))
        except Exception:
            return False
        if not name:
            return False
        lowered = name.lower()
        if not any(kw in lowered for kw in self._CONTAINER_NAME_KEYWORDS):
            return False
        try:
            return bool(om.is_object_articulable(name))
        except Exception:
            return False

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

    def _surface_geom_is_enclosed_container(
        self, model: Any, om: Any, geom_id: int
    ) -> bool:
        """True when a candidate surface geom belongs to an *enclosed* container
        appliance (fridge/oven/microwave/cabinet/drawer ...) the mobile base
        cannot maneuver into to place.

        Mirrors the pickup-side container test: the owning furniture must both
        match a container lemma (``_CONTAINER_NAME_KEYWORDS``) AND be articulable
        (have a door/drawer joint), so open shelving with a container-like name
        is not spuriously excluded. Fail-open (returns False) on any lookup error
        so incomplete data never removes a candidate.
        """
        try:
            body_id = int(model.geom_bodyid[geom_id])
            root_body_id = int(model.body_rootid[body_id])
            name = om.get_object_name(root_body_id)
        except Exception:
            return False
        if not name:
            return False
        lowered = name.lower()
        if not any(kw in lowered for kw in self._CONTAINER_NAME_KEYWORDS):
            return False
        try:
            return bool(om.is_object_articulable(name))
        except Exception:
            return False

    # --- Closed-container (fridge/cabinet/drawer) pickup exclusion ---------- #
    # Substrings identifying openable *container* furniture that can hide a
    # pickup object behind a door/drawer. The manipulation pipeline never opens
    # doors, so an object enclosed by one of these is ungraspable and must not be
    # selected as a pickup target (observed failure: reaching for an object
    # inside a closed fridge).
    _CONTAINER_NAME_KEYWORDS = (
        "fridge",
        "refrigerator",
        "freezer",
        "cabinet",
        "cupboard",
        "drawer",
        "dresser",
        "chestofdrawers",
        "nightstand",
        "wardrobe",
        "closet",
        "microwave",
        "oven",
        "dishwasher",
        "safe",
    )

    def _openable_container_boxes(
        self, env: CPUMujocoEnv
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Axis-aligned boxes of every openable container furniture in the scene.

        A container is a top-level furniture body that (a) is articulable (has a
        hinge/slide door or drawer joint) and (b) whose name matches a known
        container lemma. Returns ``(center, dims)`` AABBs so callers can test
        whether a candidate pickup object sits inside one."""
        model = env.current_model
        data = env.current_data
        om = env.object_managers[env.current_batch_index]
        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for body_id in om.top_level_bodies():
            name = om.get_object_name(body_id)
            if not name or om.is_excluded(name) or om.is_structural(name):
                continue
            lowered = name.lower()
            if not any(kw in lowered for kw in self._CONTAINER_NAME_KEYWORDS):
                continue
            try:
                if not om.is_object_articulable(name):
                    continue  # a nameless-match without a door isn't a container.
            except Exception:
                continue
            try:
                center, dims = body_aabb(model, data, int(body_id), visual_only=True)
            except Exception:
                continue
            boxes.append((np.asarray(center, dtype=np.float64), np.asarray(dims, dtype=np.float64)))
        return boxes

    def _object_is_enclosed(
        self,
        env: CPUMujocoEnv,
        obj_name: str,
        containers: list[tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        """True if ``obj_name`` sits *inside* a closed openable container.

        An object is enclosed when its centre lies within a container's XY
        footprint and below the container's top face (``top_margin`` beneath it),
        which distinguishes an object stored *inside* a fridge/cabinet from one
        resting *on top* of it (whose centre is above the top face). We never
        open doors, so such objects are ungraspable and are excluded as pickup
        targets regardless of the door's current open state."""
        if not containers:
            return False
        model = env.current_model
        data = env.current_data
        om = env.object_managers[env.current_batch_index]
        try:
            obj_center, _ = body_aabb(model, data, int(om.get_object_body_id(obj_name)))
        except Exception:
            return False
        obj_center = np.asarray(obj_center, dtype=np.float64)
        top_margin = 0.03
        xy_margin = 0.02
        for center, dims in containers:
            half = dims / 2.0
            top_z = center[2] + half[2]
            inside_xy = (
                abs(obj_center[0] - center[0]) <= half[0] + xy_margin
                and abs(obj_center[1] - center[1]) <= half[1] + xy_margin
            )
            below_top = obj_center[2] <= top_z - top_margin
            above_bottom = obj_center[2] >= center[2] - half[2] - top_margin
            if inside_xy and below_top and above_bottom:
                return True
        return False

    def _has_overhead_obstruction(self, env: CPUMujocoEnv, obj_name: str) -> bool:
        """True if another scene geom sits directly above ``obj_name`` within the
        configured clear-column, so an approach-from-above grasp would collide.

        We take the candidate's world AABB, shrink its XY footprint by
        ``overhead_clearance_xy_margin_m`` (a negative margin grows it), and scan
        every geom that does NOT belong to the candidate's own body. A geom is a
        blocker when its AABB overlaps the (shrunk) footprint by at least
        ``overhead_clearance_min_overlap_frac`` of the footprint area AND its
        bottom lies in the band ``(obj_top, obj_top + clearance_height]``. Static
        world geometry (walls, ceilings) far above is naturally excluded by the
        bounded clearance band; only genuinely overhanging clutter qualifies.
        """
        sampler_cfg = self.config.task_sampler_config
        if not getattr(sampler_cfg, "require_overhead_clearance", True):
            return False
        model = env.current_model
        data = env.current_data
        om = env.object_managers[env.current_batch_index]
        try:
            obj_body_id = int(om.get_object_body_id(obj_name))
            obj_center, obj_dims = body_aabb(model, data, obj_body_id)
        except Exception:
            return False
        obj_center = np.asarray(obj_center, dtype=np.float64)
        obj_dims = np.asarray(obj_dims, dtype=np.float64)
        obj_top = obj_center[2] + obj_dims[2] / 2.0
        clearance = float(getattr(sampler_cfg, "overhead_clearance_height_m", 0.25))
        xy_margin = float(getattr(sampler_cfg, "overhead_clearance_xy_margin_m", -0.02))
        min_frac = float(getattr(sampler_cfg, "overhead_clearance_min_overlap_frac", 0.10))

        half_x = max(obj_dims[0] / 2.0 + xy_margin, 1e-4)
        half_y = max(obj_dims[1] / 2.0 + xy_margin, 1e-4)
        foot_area = (2.0 * half_x) * (2.0 * half_y)
        o_xmin, o_xmax = obj_center[0] - half_x, obj_center[0] + half_x
        o_ymin, o_ymax = obj_center[1] - half_y, obj_center[1] + half_y

        # Geoms belonging to the candidate's own body tree (exclude from the scan).
        own_geoms = set(descendant_geoms(model, obj_body_id, visual_only=False))

        for gid in range(model.ngeom):
            if gid in own_geoms:
                continue
            if int(model.geom_bodyid[gid]) == obj_body_id:
                continue
            try:
                gc, gd = geom_aabb(model, data, [int(gid)])
            except Exception:
                continue
            g_bottom = gc[2] - gd[2] / 2.0
            # Bottom must lie in the clear column just above the object's top.
            if not (obj_top < g_bottom <= obj_top + clearance):
                continue
            g_hx, g_hy = gd[0] / 2.0, gd[1] / 2.0
            ox = max(0.0, min(o_xmax, gc[0] + g_hx) - max(o_xmin, gc[0] - g_hx))
            oy = max(0.0, min(o_ymax, gc[1] + g_hy) - max(o_ymin, gc[1] - g_hy))
            overlap = ox * oy
            if overlap >= min_frac * foot_area:
                return True
        return False

    def _get_scene_objects(self, env: CPUMujocoEnv, mass_limit: float = 100) -> list:
        """Scene pickup candidates, with objects enclosed by a closed openable
        container (fridge/cabinet/drawer) removed on top of the base filters, and
        objects with overhead obstructions (which foil top-down grasps) removed."""
        candidates = super()._get_scene_objects(env, mass_limit=mass_limit)
        containers = self._openable_container_boxes(env)
        after_enclosed = []
        dropped_enclosed = 0
        for obj in candidates:
            if containers and self._object_is_enclosed(env, obj.name, containers):
                dropped_enclosed += 1
                continue
            after_enclosed.append(obj)
        kept = []
        dropped_overhead = 0
        for obj in after_enclosed:
            if self._has_overhead_obstruction(env, obj.name):
                dropped_overhead += 1
                continue
            kept.append(obj)
        # Never exhaust the pool: if the overhead filter removed every remaining
        # candidate, fall back to the clearance-agnostic set so the house stays
        # usable (an obstructed grasp is still better than no task).
        if not kept and after_enclosed:
            log.info(
                "[MOBILE PNP] Overhead-clearance filter would empty the pickup "
                f"pool ({dropped_overhead} obstructed); keeping obstructed candidates."
            )
            kept = after_enclosed
            dropped_overhead = 0
        if dropped_enclosed or dropped_overhead:
            log.info(
                f"[MOBILE PNP] Excluded {dropped_enclosed} enclosed + "
                f"{dropped_overhead} overhead-obstructed pickup candidate(s); "
                f"{len(kept)} remain."
            )
        return kept

    # --- Direct-on-furniture / next-to place destination -------------------- #
    def _on_candidate_selected(
        self,
        env: CPUMujocoEnv,
        reference_obj_name: str,
        reference_obj_id: int,
        supporting_geom_id: int,
    ) -> bool:
        """Select the pickup + a place target, then optionally redirect the
        place target from the spawned receptacle to an existing furniture body.

        The base implementation wires a spawned receptacle (bowl) as the place
        target. With probability ``place_on_furniture_prob`` we instead point the
        place target at a modest-footprint same-room furniture body so the pickup
        object is placed directly ON it (and, when it already holds items, NEXT
        TO them). The furniture simply plays the receptacle role, so the
        downstream base-pose sampling, placement planner and success test are
        reused unchanged. If no eligible furniture is found we keep the receptacle
        wiring, so this can only add variety, never reduce yield."""
        if not super()._on_candidate_selected(
            env, reference_obj_name, reference_obj_id, supporting_geom_id
        ):
            return False
        sampler_cfg = self.config.task_sampler_config
        task_cfg = self.config.task_config
        self._verified_place_base_pose = None

        # Optionally redirect the place target to a same-room furniture surface,
        # but only *commit* the redirect when the place-feasibility probe verifies
        # the furniture is reachable AND finds a place-feasible standoff. If the
        # furniture is not verifiably reachable we revert to the spawned-receptacle
        # (bowl) wiring, so the redirect can only add variety, never reduce yield.
        prob = float(getattr(sampler_cfg, "place_on_furniture_prob", 0.0))
        broad = bool(getattr(sampler_cfg, "prefer_furniture_place", False))
        if broad:
            prob = float(getattr(sampler_cfg, "broad_furniture_place_prob", 0.85))
        if prob > 0.0 and float(np.random.random()) < prob:
            saved = (
                task_cfg.place_receptacle_name,
                task_cfg.place_target_name,
                getattr(task_cfg, "place_receptacle_start_pose", None),
            )
            if self._redirect_place_to_furniture(env, broad=broad):
                _checked, standoff, _reachable = self._find_place_standoff(
                    env, broad=broad
                )
                if standoff is not None:
                    self._verified_place_base_pose = standoff.copy()
                    # Keep the referral-expression source in lockstep with the
                    # redirected target. ``_configure_pick_and_place`` generates
                    # the place referral (hence the instruction and subtask
                    # labels) from ``self.place_receptacle_name``, but the
                    # redirect above only rewrote ``task_cfg.place_receptacle_name``.
                    # Without this sync the instruction/labels would describe the
                    # now-unused spawned bowl instead of the actual furniture
                    # destination the object is placed on.
                    self.place_receptacle_name = task_cfg.place_receptacle_name
                    log.info(
                        "[MOBILE PNP] Redirected place target to furniture "
                        f"'{task_cfg.place_receptacle_name}' (verified reachable, "
                        "direct-on-surface placement)."
                    )
                    return True
                # Not verifiably reachable -> revert to the proven receptacle path.
                (
                    task_cfg.place_receptacle_name,
                    task_cfg.place_target_name,
                    task_cfg.place_receptacle_start_pose,
                ) = saved
                log.info(
                    "[MOBILE PNP] Furniture redirect candidate not verified "
                    "reachable; reverted to spawned receptacle."
                )

        # Advisory place-reachability probe WITH receptacle-position re-sampling.
        #
        # The spawned receptacle is initially dropped at ONE sampled spot on a far
        # elevated surface. If that particular spot is not place-reachable, the
        # entire (long, ~300-670s) rollout is wasted producing a guaranteed PLACE
        # failure. So when the probe finds no reachable standoff we RE-SAMPLE the
        # receptacle to a different point (via ``_prepare_place_target``, which
        # draws a fresh spot on the surface) and re-probe, up to
        # ``place_receptacle_resample_tries`` times, keeping the FIRST spot from
        # which the arm can place. Because the sample-time probe is strictly more
        # conservative than the runtime place (probe-reachable is a subset of
        # runtime-reachable), a probe-verified spot is a high-confidence choice
        # for the runtime -- so this makes the place target reachable by
        # construction wherever a reachable spot exists on the surface.
        #
        # The probe stays *advisory*: if no spot verifies within the budget we
        # keep the last placement and do NOT reject (the conservative probe would
        # otherwise discard receptacles the runtime can actually place on and can
        # exhaust every candidate in a house). Rejection remains opt-in behind
        # ``place_reject_on_unreachable`` (default off).
        pickup_obj_name = task_cfg.pickup_obj_name
        om = env.object_managers[env.current_batch_index]
        pickup_pos = np.asarray(
            om.get_object_by_name(reference_obj_name).position, dtype=np.float64
        )
        n_resample = int(getattr(sampler_cfg, "place_receptacle_resample_tries", 8))
        checked, standoff, reachable = self._find_place_standoff(env)
        if standoff is None and n_resample > 0:
            # Snapshot the original (already-valid) receptacle placement so we can
            # RESTORE it if re-sampling fails to find a probe-verified spot. This
            # makes re-sampling strictly non-harmful: worst case we fall back to
            # the exact original placement (baseline advisory behavior); the probe
            # is more conservative than the runtime, so leaving the receptacle at
            # an arbitrary last-sampled spot could otherwise move it AWAY from a
            # spot the runtime could actually place on and regress the house.
            # Snapshot the receptacle's free-joint qpos directly (the runtime
            # object wrapper exposes no pose setter).
            model = env.current_model
            data = env.current_data
            recep_id = om.get_object_body_id(self.place_receptacle_name)
            recep_qadr = int(model.jnt_qposadr[int(model.body_jntadr[recep_id])])
            original_recep_qpos = data.qpos[recep_qadr : recep_qadr + 7].copy()
            original_start_pose = getattr(task_cfg, "place_receptacle_start_pose", None)
            tries = 0
            while standoff is None and tries < n_resample:
                tries += 1
                try:
                    if not self._prepare_place_target(
                        env,
                        self.place_receptacle_name,
                        pickup_obj_name,
                        pickup_pos,
                        supporting_geom_id,
                    ):
                        continue
                except (ValueError, ObjectPlacementError):
                    continue
                recep = om.get_object_by_name(self.place_receptacle_name)
                task_cfg.place_receptacle_start_pose = pose_mat_to_7d(recep.pose).tolist()
                checked, standoff, reachable = self._find_place_standoff(env)
            if standoff is not None:
                self._verified_place_base_pose = standoff.copy()
                log.info(
                    "[MOBILE PNP] Found a place-reachable receptacle spot for "
                    f"'{task_cfg.place_receptacle_name}' after {tries} re-sample(s)."
                )
            else:
                # No verified spot -> restore the original valid placement.
                data.qpos[recep_qadr : recep_qadr + 7] = original_recep_qpos
                mujoco.mj_forward(model, data)
                task_cfg.place_receptacle_start_pose = original_start_pose
        elif standoff is not None:
            self._verified_place_base_pose = standoff.copy()

        if not reachable and bool(
            getattr(sampler_cfg, "place_reject_on_unreachable", False)
        ):
            log.info(
                "[MOBILE PNP] Place target "
                f"'{task_cfg.place_receptacle_name}' unreachable from any "
                "standoff (centre or top-footprint grid); trying another "
                "candidate."
            )
            return False
        return True

    def _redirect_place_to_furniture(self, env: CPUMujocoEnv, broad: bool = False) -> bool:
        """Point the place target at an existing same-room furniture body,
        replacing the spawned receptacle.

        Two modes:

        * default (``broad=False``): only *small-footprint* furniture whose
          top-surface CENTRE stays within arm reach is eligible, and the object
          drops at the furniture body-origin xy (legacy centre-drop). Origin-
          under-surface and height-match guards prevent off-surface drops.

        * ``broad=True`` (``prefer_furniture_place``): *any* flat-topped furniture
          large enough to be a sensible drop surface (bed / sofa / table / counter
          / desk ...) is eligible, regardless of footprint size. This is safe
          because the runtime place builder no longer needs the centre: it falls
          back to ``_nearest_reachable_place_pose`` (a grid search over the
          furniture top for the IK-reachable point nearest the base). Eligibility
          is therefore judged by "the top has a reachable point", which the
          ``_find_place_standoff`` probe run by the caller verifies -- so the
          size/origin/height guards below are dropped for large furniture. A
          minimum half-extent floor still excludes thin ledges / chair seats.

        Returns True and rewrites ``place_receptacle_name`` / ``place_target_name``
        / ``place_receptacle_start_pose`` on success; False (leaving the receptacle
        wiring intact) when no eligible furniture exists."""
        task_cfg = self.config.task_config
        sampler_cfg = self.config.task_sampler_config
        model = env.current_model
        om = env.object_managers[env.current_batch_index]
        try:
            pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
            pickup_pos = np.asarray(pickup_obj.position, dtype=np.float64)
        except Exception:
            return False

        max_half = float(getattr(sampler_cfg, "furniture_place_max_half_extent_m", 0.40))
        min_half_broad = float(
            getattr(sampler_cfg, "broad_furniture_min_half_extent_m", 0.20)
        )
        try:
            pickup_body_id = int(om.get_object_body_id(task_cfg.pickup_obj_name))
            _pc, pickup_dims = body_aabb(model, env.current_data, pickup_body_id)
            pickup_half_xy = float(max(pickup_dims[0], pickup_dims[1]) / 2.0)
        except Exception:
            pickup_half_xy = 0.05
        # Reuse the far/same-room/elevated surface search, then map each surface
        # geom to its owning furniture body and keep the first (farthest) whose
        # footprint is small enough for its centre to stay within arm reach. In
        # broad mode, prepend directly-discovered furniture tops (bare beds /
        # tables that support nothing, invisible to the supported-object search)
        # so any flat furniture -- not just cluttered furniture -- is eligible.
        surface_geoms = self._find_far_elevated_surface_geoms(env, pickup_pos)
        if broad:
            direct_tops = self._find_furniture_top_geoms(env, pickup_pos)
            seen_g = set(surface_geoms)
            surface_geoms = direct_tops + [g for g in surface_geoms if g not in seen_g]
        pickup_root = int(model.body_rootid[int(om.get_object_body_id(task_cfg.pickup_obj_name))])
        seen_bodies: set[int] = set()
        for geom_id in surface_geoms:
            try:
                body_id = int(model.geom_bodyid[int(geom_id)])
                root_id = int(model.body_rootid[body_id])
            except Exception:
                continue
            if root_id in seen_bodies or root_id == pickup_root:
                continue
            seen_bodies.add(root_id)
            name = om.get_object_name(root_id)
            if not name or om.is_excluded(name) or om.is_structural(name):
                continue
            if broad:
                # Same furniture-type filter as ``_find_furniture_top_geoms`` so
                # chairs/stools surfaced via the supported-object search are also
                # excluded (they pass the half-extent gate but the base cannot
                # stand off from them).
                lname = name.lower()
                deny = tuple(getattr(sampler_cfg, "broad_furniture_deny_substrings", ()))
                if any(sub in lname for sub in deny):
                    continue
                if bool(getattr(sampler_cfg, "broad_furniture_type_allowlist_only", False)):
                    allow = tuple(
                        getattr(sampler_cfg, "broad_furniture_allow_substrings", ())
                    )
                    if allow and not any(sub in lname for sub in allow):
                        continue
            try:
                body_center, dims = body_aabb(model, env.current_data, root_id, visual_only=True)
            except Exception:
                continue
            try:
                surf_center, surf_dims = geom_aabb(model, env.current_data, [int(geom_id)])
            except Exception:
                continue
            if broad:
                # Broad mode: accept any flat furniture whose top surface is big
                # enough to hold the object with margin. Reachability of a top
                # point is verified by the caller's ``_find_place_standoff`` probe
                # (which reverts the redirect if unreachable), so no centre / size
                # / origin / height gate is applied here beyond a sensible floor.
                surf_half = float(min(surf_dims[0], surf_dims[1]) / 2.0)
                if surf_half < min_half_broad:
                    continue  # thin ledge / chair seat -- not a drop surface.
                margin_x = float(surf_dims[0] / 2.0 - pickup_half_xy)
                margin_y = float(surf_dims[1] / 2.0 - pickup_half_xy)
                if margin_x <= 0.0 or margin_y <= 0.0:
                    continue  # surface too small for the object footprint.
                try:
                    start_pose = pose_mat_to_7d(
                        om.get_object_by_name(name).pose
                    ).tolist()
                except Exception:
                    continue
                task_cfg.place_receptacle_name = name
                task_cfg.place_target_name = name
                task_cfg.place_receptacle_start_pose = start_pose
                log.info(
                    f"[MOBILE PNP] Broad furniture place target '{name}' "
                    f"(top {surf_dims[0]:.2f}x{surf_dims[1]:.2f}m); runtime places "
                    "at nearest reachable top point."
                )
                return True
            # Shorter footprint half-extent: the base can approach from the
            # narrow side, so the centre is reachable when the SMALLER half-extent
            # is within the gate.
            if float(min(dims[0], dims[1]) / 2.0) > max_half:
                continue
            try:
                furniture = om.get_object_by_name(name)
            except Exception:
                continue
            # Stability guard: the placement planner drops the carried object at
            # the receptacle *body-origin* xy and at the body-AABB top z. A stable
            # rest therefore requires (a) the body-origin xy to sit inside the
            # chosen top-surface geom footprint (shrunk by the object footprint so
            # it does not overhang) and (b) the body-AABB top to coincide with
            # that surface's top (so the object lands ON the surface, not floating
            # above a shorter surface of a taller body). Furniture whose origin is
            # not under its top surface (e.g. an L-shaped stand) is skipped so it
            # cannot produce off-surface drops.
            origin_xy = np.asarray(furniture.position, dtype=np.float64)[:2]
            surf_xy = np.asarray(surf_center[:2], dtype=np.float64)
            margin_x = float(surf_dims[0] / 2.0 - pickup_half_xy)
            margin_y = float(surf_dims[1] / 2.0 - pickup_half_xy)
            if margin_x <= 0.0 or margin_y <= 0.0:
                continue  # surface too small for the object footprint.
            if (
                abs(origin_xy[0] - surf_xy[0]) > margin_x
                or abs(origin_xy[1] - surf_xy[1]) > margin_y
            ):
                continue  # body origin overhangs the top surface -> would drop.
            body_top_z = float(body_center[2] + dims[2] / 2.0)
            surf_top_z = float(surf_center[2] + surf_dims[2] / 2.0)
            if abs(body_top_z - surf_top_z) > 0.08:
                continue  # placement height would not match the chosen surface.
            try:
                start_pose = pose_mat_to_7d(furniture.pose).tolist()
            except Exception:
                continue
            task_cfg.place_receptacle_name = name
            task_cfg.place_target_name = name
            task_cfg.place_receptacle_start_pose = start_pose
            return True
        return False

    def _find_place_standoff(
        self, env: CPUMujocoEnv, broad: bool = False
    ) -> tuple[bool, np.ndarray | None, bool]:
        """Probe for a base standoff from which the carried pickup object can be
        placed on the receptacle.

        Returns ``(checked, pose)``:

        Returns ``(checked, record_pose, reachable)``:

        * ``record_pose`` is a base-locked standoff (4x4 world pose) from which
          the receptacle *centre* place pose is IK-feasible, or ``None``. When
          non-None the caller records it as the place nav goal so navigation
          parks at a place-feasible pose and the PLACE phase builds in place (no
          ring search, no teleport). It is CENTRE-only on purpose: recording a
          grid-derived off-centre standoff would shift the parked base and can
          regress a previously-succeeding episode.
        * ``reachable`` is True when *any* standoff can place the object on the
          receptacle -- at the centre OR (mirroring the runtime
          ``_nearest_reachable_place_pose`` fallback) at a nearest-to-base point
          on the receptacle top footprint grid. This is the signal the optional
          ``place_reject_on_unreachable`` path uses, so it matches the runtime's
          true reachability rather than the stricter centre-only test.
        * ``checked`` is False when the probe could not run (verification
          disabled, or grasps / metadata / kinematics unavailable); in that case
          ``reachable`` is True (fail-open: incomplete data never blocks
          sampling).

        Mirrors the placement planner's pose construction (the carried-object
        grasp orientation, translated so the *object* lands on the receptacle
        top), base-locked to match the runtime manip phase.
        """
        sampler_cfg = self.config.task_sampler_config
        task_cfg = self.config.task_config
        if not getattr(sampler_cfg, "verify_place_reachable", True):
            return (False, None, True)

        om = env.object_managers[env.current_batch_index]
        model = env.current_model
        data = env.current_data
        robot_view = env.current_robot.robot_view
        try:
            kinematics = env.current_robot.kinematics
        except Exception:
            return (False, None, True)  # no kinematics available -> cannot probe.

        try:
            pickup = om.get_object_by_name(task_cfg.pickup_obj_name)
            receptacle = om.get_object_by_name(task_cfg.place_receptacle_name)
            receptacle_id = om.get_object_body_id(task_cfg.place_receptacle_name)
            pickup_id = om.get_object_body_id(task_cfg.pickup_obj_name)
        except Exception:
            return (False, None, True)

        # Carried-grasp orientations to probe: the same cached grasps the pickup
        # loop validated as non-colliding, expressed in world frame.
        asset_uid = self.get_asset_uid_from_object(env, task_cfg.pickup_obj_name)
        if not asset_uid:
            return (False, None, True)
        try:
            _gripper, cached_grasps = load_grasps_for_object(asset_uid, 512)
        except (KeyError, ValueError):
            return (False, None, True)
        if cached_grasps is None or len(cached_grasps) == 0:
            return (False, None, True)
        object_pose = pos_quat_to_pose_mat(pickup.position, pickup.quat)
        grasp_poses_world = object_pose @ cached_grasps
        try:
            noncolliding = get_noncolliding_grasp_mask(model, data, grasp_poses_world, 64)
            grasp_poses_world = grasp_poses_world[np.asarray(noncolliding, dtype=bool)]
        except (KeyError, ValueError):
            pass  # keep all grasps if the collision bodies are absent.
        if len(grasp_poses_world) == 0:
            return (False, None, True)
        max_grasps = int(getattr(sampler_cfg, "place_reachable_max_grasps", 8))
        if broad:
            # Broad furniture probes many base gaps/anchors; cap grasps tighter to
            # keep the IK-call count (and sampling time) bounded.
            max_grasps = min(
                max_grasps,
                int(getattr(sampler_cfg, "broad_place_reachable_max_grasps", 4)),
            )
        grasp_poses_world = grasp_poses_world[:max_grasps]

        # Placement-pose geometry (planner formula, see
        # ``pick_and_place_planner_policy._get_placement_poses``).
        try:
            rec_center, rec_size = body_aabb(model, data, receptacle_id)
            pick_center, pick_size = body_aabb(model, data, pickup_id)
        except Exception:
            return (False, None, True)
        receptacle_top_z = float(rec_center[2] + rec_size[2] / 2.0)
        pickup_bottom_z = float(pick_center[2] - pick_size[2] / 2.0)
        z_off = float(getattr(sampler_cfg, "place_reachable_z_offset_m", 0.05))
        rec_xy = np.asarray(receptacle.position, dtype=np.float64)[:2]
        pickup_pos = np.asarray(pickup.position, dtype=np.float64)

        # Runtime place fallback replication: when the receptacle-CENTRE place
        # pose is not IK-reachable, the FSM does NOT give up -- it searches a
        # grid over the receptacle top footprint (shrunk by the object half-
        # extent + margin) for the nearest-to-base reachable point and places
        # there (see ``_nearest_reachable_place_pose`` /
        # ``_get_placement_poses`` in the policy). A probe that tests only the
        # centre therefore under-reports reachability and false-rejects
        # receptacles the runtime can place on. Mirror the runtime grid here so
        # the probe's verdict matches the runtime, making rejection / re-sampling
        # safe to enable. Config knobs mirror the policy's
        # ``place_edge_margin_m`` / ``place_search_grid_n``.
        grid_margin = float(getattr(sampler_cfg, "place_reachable_edge_margin_m", 0.03))
        grid_n = int(getattr(sampler_cfg, "place_reachable_search_grid_n", 5))
        if broad:
            grid_n = min(
                grid_n, int(getattr(sampler_cfg, "broad_place_reachable_grid_n", 3))
            )
        half_x = max(float(rec_size[0] / 2.0 - pick_size[0] / 2.0 - grid_margin), 0.0)
        half_y = max(float(rec_size[1] / 2.0 - pick_size[1] / 2.0 - grid_margin), 0.0)
        grid_xs = np.linspace(-half_x, half_x, grid_n) + float(rec_center[0])
        grid_ys = np.linspace(-half_y, half_y, grid_n) + float(rec_center[1])

        def place_xy_candidates(base_pose: np.ndarray) -> list[np.ndarray]:
            """Receptacle-top XY place points, receptacle centre first, then the
            footprint grid ordered nearest-to-base (mirroring the runtime)."""
            base_xy = base_pose[:2, 3]
            grid = [
                np.array([x, y], dtype=np.float64)
                for x in grid_xs
                for y in grid_ys
            ]
            grid.sort(key=lambda p: float((p[0] - base_xy[0]) ** 2 + (p[1] - base_xy[1]) ** 2))
            return [rec_xy] + grid

        def place_poses_for(grasp_world: np.ndarray, place_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            clearance = max(float(grasp_world[2, 3]) - pickup_bottom_z, 0.0)
            preplace = grasp_world.copy()
            preplace[:2, 3] = place_xy
            preplace[2, 3] = receptacle_top_z + clearance + z_off
            preplace[:3, 3] += grasp_world[:3, 3] - pickup_pos
            place = preplace.copy()
            place[2, 3] = receptacle_top_z + clearance
            return preplace, place

        base_z = float(getattr(sampler_cfg, "mobile_base_z", 0.1))
        gripper_mg_ids = robot_view.get_gripper_movegroup_ids()
        # Lock the base: solve IK for arm/gripper joints only, with the base held
        # at the probed standoff. Including the base move group would let the IK
        # solver slide the (holonomic) base in x/y/yaw to reach the target,
        # trivially satisfying every pose and defeating the reachability test. The
        # runtime manip phase is base-locked, so we must probe base-locked too.
        unlocked = [
            mg for mg in robot_view.move_group_ids() if mg != _BASE_MG_ID_SAMPLER
        ]

        original_base = robot_view.base.pose.copy()
        original_qpos = robot_view.get_qpos_dict()
        namespace = getattr(robot_view, "namespace", "robot_0/")

        def ik_ok(base_pose: np.ndarray, target: np.ndarray) -> bool:
            for mg in gripper_mg_ids:
                try:
                    jp = kinematics.ik(
                        mg, target, unlocked, original_qpos, base_pose, max_iter=150
                    )
                except Exception:
                    jp = None
                if jp is not None:
                    return True
            return False

        # Deterministic ring of receptacle-facing standoffs (reuse the pickup
        # standoff band). A *deterministic* ring is used deliberately rather than
        # the runtime's stochastic ``place_robot_near`` sampler: sampling here
        # would consume the global RNG and perturb all downstream episode sampling
        # (grasp/object/pose selection), breaking reproducibility and changing
        # outcomes. The ring is only used to *record* a verified place nav goal
        # (advisory); it never rejects, so incompleteness merely means no recorded
        # standoff (the runtime then falls back to its own hint + ring search).
        r_lo, r_hi = getattr(sampler_cfg, "manip_standoff_radius_range", (0.35, 0.7))
        radii = np.linspace(float(r_lo), float(r_hi), 2)
        n_ang = int(getattr(sampler_cfg, "place_reachable_standoff_angles", 12))
        found_pose: np.ndarray | None = None
        # ``found_pose`` is the recorded (centre-feasible) nav goal, preserving
        # prior behaviour exactly. ``reachable_pose`` additionally counts
        # grid-feasible (off-centre) standoffs -- used ONLY to judge whether the
        # receptacle is reachable at all (for the rejection path), never recorded
        # as the nav goal. The off-centre grid probe is GATED on rejection being
        # enabled (its extra IK calls perturb the shared RNG); when off, the
        # receptacle is treated as reachable iff a centre standoff was found and
        # the probe is a strict no-op vs the prior centre-only behaviour.
        grid_reject_enabled = bool(
            getattr(sampler_cfg, "place_reject_on_unreachable", False)
        )
        # Broad furniture mode: the runtime places at the nearest reachable point
        # on a (potentially large) furniture top, so a centre-only probe wrongly
        # rejects big surfaces whose centre is out of arm reach. Enable the grid
        # probe AND allow a grid-feasible standoff to be RECORDED as the nav goal
        # (so navigation parks where an off-centre top point is reachable). This
        # is what makes bare beds / large tables usable as place targets.
        grid_enabled = grid_reject_enabled or broad
        reachable_pose: np.ndarray | None = None
        # Standoff anchor points: the ring of base standoffs is built around each
        # anchor at radius ``manip_standoff_radius_range`` facing the anchor. For
        # normal (small) receptacles the single anchor is the receptacle centre
        # (unchanged behaviour). For BROAD furniture the centre of a large top is
        # unreachable from any collision-free standoff (the base would sit inside
        # the furniture footprint), so we anchor on EDGE points of the top and put
        # the base OUTSIDE the footprint (beyond the edge) reaching inward -- this
        # mirrors the runtime, which parks beside the furniture and places at the
        # nearest reachable top point.
        broad_specs: list[tuple[np.ndarray, np.ndarray]] = []  # (anchor_xy, base_xy)
        if broad:
            surf_half_x = float(rec_size[0] / 2.0)
            surf_half_y = float(rec_size[1] / 2.0)
            edge_reach = float(getattr(sampler_cfg, "broad_place_edge_reach_m", 0.35))
            base_gap = float(getattr(sampler_cfg, "broad_place_base_gap_m", 0.55))
            # Edge anchors: midpoints of the 4 sides, plus their halves, inset by
            # the object footprint so the object rests fully on the surface.
            for frac in (-0.5, 0.0, 0.5):
                # +X / -X edges (base stands off in ±x beyond the edge)
                ax = float(rec_center[0] + surf_half_x - edge_reach)
                ay = float(rec_center[1] + frac * 2.0 * (surf_half_y - edge_reach))
                broad_specs.append((np.array([ax, ay]),
                                    np.array([rec_center[0] + surf_half_x + base_gap, ay])))
                ax2 = float(rec_center[0] - surf_half_x + edge_reach)
                broad_specs.append((np.array([ax2, ay]),
                                    np.array([rec_center[0] - surf_half_x - base_gap, ay])))
                # +Y / -Y edges
                bx = float(rec_center[0] + frac * 2.0 * (surf_half_x - edge_reach))
                by = float(rec_center[1] + surf_half_y - edge_reach)
                broad_specs.append((np.array([bx, by]),
                                    np.array([bx, rec_center[1] + surf_half_y + base_gap])))
                by2 = float(rec_center[1] - surf_half_y + edge_reach)
                broad_specs.append((np.array([bx, by2]),
                                    np.array([bx, rec_center[1] - surf_half_y - base_gap])))
            anchors = [a for a, _ in broad_specs]
            if len(anchors) == 0:
                anchors = [rec_xy]
                broad_specs = [(rec_xy, None)]
        else:
            anchors = [rec_xy]
            broad_specs = [(rec_xy, None)]
        # Broad furniture: sweep the base along the outward normal at a set of
        # REACH-APPROPRIATE gaps (measured beyond the edge anchor), plus a small
        # lateral jitter, instead of the previous ``anchor + outward*r*2`` which
        # placed the base up to 1.4m from the anchor (well beyond arm reach) and
        # tried only 2 poses per edge. The nominal gap comes from
        # ``broad_place_base_gap_m``; we probe a few gaps around it so the base
        # parks close enough to reach the top point while staying outside the
        # footprint / clear of collisions.
        base_gap_nom = float(getattr(sampler_cfg, "broad_place_base_gap_m", 0.55))
        # Gaps span from the nominal edge gap outward: larger gaps let the stowed
        # arm (fr3_link0 projects toward the surface) clear the furniture body,
        # while the runtime still places at the nearest reachable top point, so a
        # near-edge point on a large top stays reachable even from a larger gap.
        broad_gaps = [
            max(base_gap_nom + d, 0.12)
            for d in (-0.40, -0.20, 0.0)
        ]
        broad_lat = [0.0]
        try:
            for anchor_xy, fixed_base_xy in broad_specs:
                if found_pose is not None:
                    break
                for r in radii:
                    for k in range(n_ang):
                        if broad and fixed_base_xy is not None:
                            # Broad furniture: base parks OUTSIDE the footprint
                            # beyond the chosen edge, facing inward. ``k`` indexes
                            # a grid over outward gap (``broad_gaps``) and lateral
                            # offset (``broad_lat``) so the base lands at a
                            # reachable, collision-free spot near the edge. ``r``
                            # is unused here (the gap replaces it); the outer
                            # radius loop is short-circuited below.
                            outward_vec = fixed_base_xy - anchor_xy
                            nrm = float(np.linalg.norm(outward_vec)) or 1.0
                            outward = outward_vec / nrm
                            lateral = np.array([-outward[1], outward[0]])
                            if k >= len(broad_gaps) * len(broad_lat):
                                continue
                            gi = k % len(broad_gaps)
                            li = k // len(broad_gaps)
                            gap = broad_gaps[gi]
                            lat = broad_lat[li]
                            # ``fixed_base_xy`` sits at (true edge + base_gap_nom)
                            # along ``outward`` from the inset anchor, so the anchor
                            # -> edge distance is ``nrm - base_gap_nom``. Place the
                            # base at (edge + gap) => anchor + (edge_dist + gap) so
                            # the base is genuinely OUTSIDE the footprint. The prior
                            # ``anchor + outward*gap`` measured the gap from the
                            # inset anchor and left the base INSIDE the footprint
                            # for large tops -- the dominant "base vs furniture"
                            # collision that reverted almost every redirect.
                            edge_dist = max(nrm - base_gap_nom, 0.0)
                            reach_out = edge_dist + gap
                            bx = float(anchor_xy[0] + outward[0] * reach_out + lateral[0] * lat)
                            by = float(anchor_xy[1] + outward[1] * reach_out + lateral[1] * lat)
                        else:
                            a = 2.0 * np.pi * k / max(n_ang, 1)
                            bx = float(anchor_xy[0] + r * np.cos(a))
                            by = float(anchor_xy[1] + r * np.sin(a))
                        theta = float(np.arctan2(anchor_xy[1] - by, anchor_xy[0] - bx))
                        c, s = np.cos(theta), np.sin(theta)
                        base_pose = np.eye(4)
                        base_pose[:3, :3] = np.array(
                            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
                        )
                        base_pose[0, 3], base_pose[1, 3], base_pose[2, 3] = bx, by, base_z
                        # Skip standoffs that collide with the environment.
                        robot_view.base.pose = base_pose
                        mujoco.mj_forward(model, data)
                        try:
                            if env.check_robot_collision_in_current_pose(namespace):
                                continue
                        except Exception:
                            pass
                        place_xys = (
                            place_xy_candidates(base_pose) if grid_enabled else []
                        )
                        # Primary place point: receptacle CENTRE for normal
                        # receptacles (unchanged), or the current anchor top point
                        # for broad furniture (the runtime places at the nearest
                        # reachable top point, which is near an edge, not centre).
                        primary_xy = anchor_xy if broad else rec_xy
                        for grasp_world in grasp_poses_world:
                            preplace_c, place_c = place_poses_for(grasp_world, primary_xy)
                            if ik_ok(base_pose, place_c) and ik_ok(base_pose, preplace_c):
                                found_pose = base_pose.copy()
                                reachable_pose = found_pose
                                break
                            # Runtime ``_nearest_reachable_place_pose`` mirror: the
                            # runtime can place off-centre on the top footprint when
                            # the centre is unreachable. Track that as REACHABLE (for
                            # the opt-in rejection path) WITHOUT recording it as the
                            # nav goal. GATED on rejection being enabled: the extra
                            # grid IK calls advance the IK solver's RNG and would
                            # perturb all downstream episode sampling
                            # (``place_robot_near`` etc.), so when rejection is off
                            # (default) we skip the grid entirely and keep the exact
                            # baseline IK-call sequence. In ``broad`` mode the
                            # grid-feasible standoff is ALSO recorded as the nav goal.
                            if grid_enabled and reachable_pose is None:
                                for place_xy in place_xys[1:]:
                                    preplace, place = place_poses_for(grasp_world, place_xy)
                                    if ik_ok(base_pose, place) and ik_ok(base_pose, preplace):
                                        reachable_pose = base_pose.copy()
                                        if broad and found_pose is None:
                                            found_pose = base_pose.copy()
                                        break
                        if found_pose is not None:
                            break
                    if found_pose is not None:
                        break
                    if broad and fixed_base_xy is not None:
                        # Gap/lateral grid already swept via ``k``; the outer
                        # radius loop would only re-probe identical poses.
                        break
                if found_pose is not None:
                    break
        finally:
            robot_view.base.pose = original_base
            robot_view.set_qpos_dict(original_qpos)
            mujoco.mj_forward(model, data)
        # When the off-centre grid is disabled we never reject, so report
        # reachable=True unconditionally (rejection is gated on this flag AND on
        # ``place_reject_on_unreachable``). When enabled, reachability reflects
        # whether any centre-or-grid standoff could place the object.
        reachable = True if not grid_enabled else (reachable_pose is not None)
        return (True, found_pose, reachable)

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
