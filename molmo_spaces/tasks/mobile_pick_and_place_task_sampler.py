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
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, geom_aabb
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

        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

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

    def _get_scene_objects(self, env: CPUMujocoEnv, mass_limit: float = 100) -> list:
        """Scene pickup candidates, with objects enclosed by a closed openable
        container (fridge/cabinet/drawer) removed on top of the base filters."""
        candidates = super()._get_scene_objects(env, mass_limit=mass_limit)
        containers = self._openable_container_boxes(env)
        if not containers:
            return candidates
        kept = []
        dropped = 0
        for obj in candidates:
            if self._object_is_enclosed(env, obj.name, containers):
                dropped += 1
                continue
            kept.append(obj)
        if dropped:
            log.info(
                f"[MOBILE PNP] Excluded {dropped} pickup candidate(s) enclosed by a "
                f"closed openable container; {len(kept)} remain."
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
        if prob > 0.0 and float(np.random.random()) < prob:
            saved = (
                task_cfg.place_receptacle_name,
                task_cfg.place_target_name,
                getattr(task_cfg, "place_receptacle_start_pose", None),
            )
            if self._redirect_place_to_furniture(env):
                _checked, standoff = self._find_place_standoff(env)
                if standoff is not None:
                    self._verified_place_base_pose = standoff.copy()
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
        checked, standoff = self._find_place_standoff(env)
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
                checked, standoff = self._find_place_standoff(env)
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

        if standoff is None and bool(
            getattr(sampler_cfg, "place_reject_on_unreachable", False)
        ):
            log.info(
                "[MOBILE PNP] Place target "
                f"'{task_cfg.place_receptacle_name}' unreachable from any "
                "standoff; trying another candidate."
            )
            return False
        return True

    def _redirect_place_to_furniture(self, env: CPUMujocoEnv) -> bool:
        """Point the place target at an existing same-room furniture body whose
        top-surface centre is reachable, replacing the spawned receptacle.

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
        try:
            pickup_body_id = int(om.get_object_body_id(task_cfg.pickup_obj_name))
            _pc, pickup_dims = body_aabb(model, env.current_data, pickup_body_id)
            pickup_half_xy = float(max(pickup_dims[0], pickup_dims[1]) / 2.0)
        except Exception:
            pickup_half_xy = 0.05
        # Reuse the far/same-room/elevated surface search, then map each surface
        # geom to its owning furniture body and keep the first (farthest) whose
        # footprint is small enough for its centre to stay within arm reach.
        surface_geoms = self._find_far_elevated_surface_geoms(env, pickup_pos)
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
            try:
                body_center, dims = body_aabb(model, env.current_data, root_id, visual_only=True)
            except Exception:
                continue
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
            try:
                surf_center, surf_dims = geom_aabb(model, env.current_data, [int(geom_id)])
            except Exception:
                continue
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

    def _find_place_standoff(self, env: CPUMujocoEnv) -> tuple[bool, np.ndarray | None]:
        """Probe for a base standoff from which the carried pickup object can be
        placed on the receptacle.

        Returns ``(checked, pose)``:

        * ``(False, None)`` -- the probe could not run (verification disabled, or
          grasps / metadata / kinematics unavailable). The caller keeps the
          candidate and records no verified pose (fail-open: incomplete data never
          blocks sampling).
        * ``(True, None)`` -- the probe ran and NO standoff is place-feasible. The
          caller rejects the candidate (this receptacle is genuinely unreachable).
        * ``(True, pose)`` -- a base-locked place-feasible standoff (4x4 world
          pose). The caller keeps the candidate and records the pose as the place
          nav goal, so navigation parks at a place-feasible pose and the PLACE
          phase builds in place (no ring search, no teleport).

        Mirrors the placement planner's pose construction (the carried-object
        grasp orientation, translated so the *object* lands on the receptacle
        top), base-locked to match the runtime manip phase.
        """
        sampler_cfg = self.config.task_sampler_config
        task_cfg = self.config.task_config
        if not getattr(sampler_cfg, "verify_place_reachable", True):
            return (False, None)

        om = env.object_managers[env.current_batch_index]
        model = env.current_model
        data = env.current_data
        robot_view = env.current_robot.robot_view
        try:
            kinematics = env.current_robot.kinematics
        except Exception:
            return (False, None)  # no kinematics available -> cannot probe.

        try:
            pickup = om.get_object_by_name(task_cfg.pickup_obj_name)
            receptacle = om.get_object_by_name(task_cfg.place_receptacle_name)
            receptacle_id = om.get_object_body_id(task_cfg.place_receptacle_name)
            pickup_id = om.get_object_body_id(task_cfg.pickup_obj_name)
        except Exception:
            return (False, None)

        # Carried-grasp orientations to probe: the same cached grasps the pickup
        # loop validated as non-colliding, expressed in world frame.
        asset_uid = self.get_asset_uid_from_object(env, task_cfg.pickup_obj_name)
        if not asset_uid:
            return (False, None)
        try:
            _gripper, cached_grasps = load_grasps_for_object(asset_uid, 512)
        except (KeyError, ValueError):
            return (False, None)
        if cached_grasps is None or len(cached_grasps) == 0:
            return (False, None)
        object_pose = pos_quat_to_pose_mat(pickup.position, pickup.quat)
        grasp_poses_world = object_pose @ cached_grasps
        try:
            noncolliding = get_noncolliding_grasp_mask(model, data, grasp_poses_world, 64)
            grasp_poses_world = grasp_poses_world[np.asarray(noncolliding, dtype=bool)]
        except (KeyError, ValueError):
            pass  # keep all grasps if the collision bodies are absent.
        if len(grasp_poses_world) == 0:
            return (False, None)
        max_grasps = int(getattr(sampler_cfg, "place_reachable_max_grasps", 8))
        grasp_poses_world = grasp_poses_world[:max_grasps]

        # Placement-pose geometry (planner formula, see
        # ``pick_and_place_planner_policy._get_placement_poses``).
        try:
            rec_center, rec_size = body_aabb(model, data, receptacle_id)
            pick_center, pick_size = body_aabb(model, data, pickup_id)
        except Exception:
            return (False, None)
        receptacle_top_z = float(rec_center[2] + rec_size[2] / 2.0)
        pickup_bottom_z = float(pick_center[2] - pick_size[2] / 2.0)
        z_off = float(getattr(sampler_cfg, "place_reachable_z_offset_m", 0.05))
        rec_xy = np.asarray(receptacle.position, dtype=np.float64)[:2]
        pickup_pos = np.asarray(pickup.position, dtype=np.float64)

        def place_poses_for(grasp_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            clearance = max(float(grasp_world[2, 3]) - pickup_bottom_z, 0.0)
            preplace = grasp_world.copy()
            preplace[:2, 3] = rec_xy
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
        try:
            for r in radii:
                for k in range(n_ang):
                    a = 2.0 * np.pi * k / max(n_ang, 1)
                    bx = float(rec_xy[0] + r * np.cos(a))
                    by = float(rec_xy[1] + r * np.sin(a))
                    theta = float(np.arctan2(rec_xy[1] - by, rec_xy[0] - bx))
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
                    for grasp_world in grasp_poses_world:
                        preplace, place = place_poses_for(grasp_world)
                        # Place (lower) is the binding constraint; test it first as
                        # a cheap reject, then confirm the pre-place approach.
                        if ik_ok(base_pose, place) and ik_ok(base_pose, preplace):
                            found_pose = base_pose.copy()
                            break
                    if found_pose is not None:
                        break
                if found_pose is not None:
                    break
        finally:
            robot_view.base.pose = original_base
            robot_view.set_qpos_dict(original_qpos)
            mujoco.mj_forward(model, data)
        return (True, found_pose)

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
