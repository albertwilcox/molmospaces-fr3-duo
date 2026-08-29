import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from molmo_spaces.env.arena.arena_utils import modify_mjmodel_thor_articulated
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.task_sampler import (
    BaseMujocoTaskSampler,
)
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask, RobotPlacementError
from molmo_spaces.utils.pose import pose_mat_to_7d

if TYPE_CHECKING:
    from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig


log = logging.getLogger(__name__)


class RolloutFailure(Exception):
    """Exception for when scene setup fails."""

    pass


class NavToObjTaskSampler(BaseMujocoTaskSampler):
    """
    Default task sampler for RBY1 navigation to object tasks with house iteration control.
    House order (`house_inds`) and samples per house are provided via config.
    """

    def __init__(self, config: "NavToObjBaseConfig") -> None:
        super().__init__(config)
        self.candidate_objects: None | list[MlSpacesObject] = None
        self._task_counter = None  # Track tasks within the same house for variety
        self._cached_thormap = None  # Cache occupancy map per house

        # If pickup_types is None, default to empty list which matches any object type.
        # Objects are then filtered by navigability and visibility in _get_scene_objects().
        if config.task_sampler_config.pickup_types is None:
            config.task_sampler_config.pickup_types = []

    def init_scene(self, env) -> None:
        # Initialize base randomizers (texture, lighting, dynamics)
        super().init_scene(env)

        log.info(
            f"Setting up scene for house {self.current_house_index}, task {self._task_counter}..."
        )
        model = env.mj_model
        data = env.mj_datas[0]
        modify_mjmodel_thor_articulated(model, data)

        # New house - reset counters
        self._task_counter = 0
        log.debug(f"New house {self.current_house_index} - resetting object tracking")

        # Generate occupancy map ONCE per house (for A* planner)
        import gc

        from molmo_spaces.utils.scene_maps import ProcTHORMap

        if self._cached_thormap is not None:
            del self._cached_thormap
            gc.collect()

        log.info(f"Generating occupancy map for house {self.current_house_index}")
        self._cached_thormap = ProcTHORMap.from_mj_model_path(
            model_path=env.current_model_path,
            agent_radius=self.config.task_sampler_config.robot_safety_radius,
            px_per_m=200,
            device_id=None,
        )
        log.info("Occupancy map generated successfully")

        candidate_objects = self._get_scene_objects(env)
        candidate_objects = self.balance_sample_names(candidate_objects)
        # Shuffle order deterministically per house/task for variety
        np.random.shuffle(candidate_objects)
        self.candidate_objects = candidate_objects

        np.random.shuffle(self.candidate_objects)

    def randomize_scene(self, env: CPUMujocoEnv, robot_view) -> None:
        """Setup scene state: robot joints, texture randomization, cameras."""
        # randomize scene here
        super().randomize_scene(env, robot_view)

        model = env.current_model
        data = env.current_data
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Set robot joints
        for group_name, qpos in self.config.robot_config.init_qpos.items():
            qpos = np.array(qpos)
            noise_range = self.config.robot_config.init_qpos_noise_range
            if noise_range is not None and group_name in noise_range:
                noise_mag = np.array(noise_range[group_name])
                perturb = np.random.uniform(-noise_mag, noise_mag)
            else:
                perturb = np.zeros_like(qpos)
            robot_view.get_move_group(group_name).joint_pos = qpos + perturb

        # Reset controllers to hold current positions (important for torso/head)
        for robot in env.robots:
            for controller in robot.controllers.values():
                controller.reset()

        log.info("Scene setup completed.\n")

    def resolve_visibility_object(self, env: CPUMujocoEnv, key: str) -> str | None:
        """Resolve special visibility object keys.

        Handles:
        - __pickup_object__: Current pickup/nav object (returns first candidate instance)
        """
        if key == "__pickup_object__":
            # Return the first candidate object instance for visibility checking
            if (
                hasattr(self.config.task_config, "pickup_obj_candidates")
                and self.config.task_config.pickup_obj_candidates
                and len(self.config.task_config.pickup_obj_candidates) > 0
            ):
                return self.config.task_config.pickup_obj_candidates[0]
            elif (
                hasattr(self.config.task_config, "pickup_obj_name")
                and self.config.task_config.pickup_obj_name
            ):
                return self.config.task_config.pickup_obj_name
            return None

        # Delegate to base class for other keys (e.g., __gripper__)
        return super().resolve_visibility_object(env, key)

    def _sample_task(self, env: CPUMujocoEnv) -> NavToObjTask:
        """Sample a navigation to object task configuration and create the task."""
        # Set current batch index to 0 (most common case for single-batch environments)
        # TODO(rose) at some point: handle multi-batch environments properly
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        # Get ObjectManager for type extraction
        om = env.object_managers[env.current_batch_index]

        keep_task_cfg = self.config.task_config.pickup_obj_name is not None

        excluded_types = set()

        unique_objects = set(self.candidate_objects)

        sample_success = False
        num_attempts_left = len(self.candidate_objects)
        while not sample_success and num_attempts_left > 0:
            num_attempts_left -= 1

            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_select_object")

            if not keep_task_cfg:
                self.config.task_config.pickup_obj_name = None

            # Sample nav object
            if self.config.task_config.pickup_obj_name is None:
                object_index = self._task_counter % len(self.candidate_objects)
                selected_obj = self.candidate_objects[object_index]

                # Extract object type (e.g., "lettuce" from "lettuce_58a2d909...")
                pickup_obj_type = om.fallback_expression(selected_obj.name)

                if pickup_obj_type in excluded_types:
                    continue

                # Get synset for semantic information
                synset = om.get_annotation_synset(selected_obj.name)

                # Collect all objects of this type in the scene
                same_type_candidates = [
                    obj.name
                    for obj in unique_objects
                    if om.fallback_expression(obj.name) == pickup_obj_type
                ]

                if len(same_type_candidates) > self.config.task_sampler_config.max_valid_candidates:
                    log.info(
                        f"Skipping {pickup_obj_type} with {len(same_type_candidates)} instances in scene."
                    )
                    excluded_types.add(pickup_obj_type)
                    continue

                # Set the instance name and store all candidates
                self.config.task_config.pickup_obj_name = selected_obj.name
                self.config.task_config.pickup_obj_candidates = same_type_candidates

                # Store semantic category for eval reconstruction
                self.config.task_config.pickup_obj_category = pickup_obj_type
                self.config.task_config.pickup_obj_synset = synset

                log.info(
                    f"[OK] Attempting object type '{pickup_obj_type}' (synset: {synset}) with {len(same_type_candidates)} instances: {same_type_candidates}"
                )
                log.info(f"Selected initial instance for robot placement: {selected_obj.name}")
            else:
                # If pickup_obj_name is pre-specified, it might be a type or specific instance
                # Try to interpret it as a type and collect candidates
                if self.config.task_config.pickup_obj_candidates is None:
                    pickup_obj_type = om.category_from_name(self.config.task_config.pickup_obj_name)
                    synset = om.get_annotation_synset(self.config.task_config.pickup_obj_name)
                    same_type_candidates = [
                        obj.name
                        for obj in self.candidate_objects
                        if om.fallback_expression(obj.name) == pickup_obj_type
                    ]

                    if len(same_type_candidates) == 0:
                        # Might be a specific instance name, use it as-is
                        same_type_candidates = [self.config.task_config.pickup_obj_name]
                        log.info(
                            f"[OK] Using pre-specified object instance: {self.config.task_config.pickup_obj_name}"
                        )
                    else:
                        self.config.task_config.pickup_obj_candidates = same_type_candidates
                        log.info(
                            f"[OK] Using pre-specified object type '{pickup_obj_type}' (synset: {synset}) with {len(same_type_candidates)} instances"
                        )

                    # Store semantic category
                    self.config.task_config.pickup_obj_category = pickup_obj_type
                    self.config.task_config.pickup_obj_synset = synset
                else:
                    log.info("[OK] Using pre-configured pickup_obj_name and pickup_obj_candidates")
                    # If not set, try to infer from existing pickup_obj_name
                    if self.config.task_config.pickup_obj_category is None:
                        self.config.task_config.pickup_obj_category = om.category_from_name(
                            self.config.task_config.pickup_obj_name
                        )
                        self.config.task_config.pickup_obj_synset = om.get_annotation_synset(
                            self.config.task_config.pickup_obj_name
                        )

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_select_object")

            self._task_counter += 1  # update counter, so we don't re-try same object

            try:
                self._sample_and_place_robot(env)
            except RobotPlacementError:
                log.exception("Caught when attempting to place robot. Retrying")
                continue

            # Ensure robot is in final position before camera setup
            mujoco.mj_forward(env.current_model, env.current_data)

            # Setup cameras after navigation object and robot placement
            # This allows cameras to use task-specific info (navigation object)
            self.setup_cameras(env)

            sample_success = True
            break

        if not sample_success:
            raise HouseInvalidForTask(
                f"Unable to sample a valid task out of {len(self.candidate_objects)} candidate objects"
            )

        # Here we just copy the ObjNavTask target name for completeness, even if unused
        pickup_obj_name = self.config.task_config.pickup_obj_name

        # Get natural name if available
        try:
            object_name = om.fallback_expression(pickup_obj_name)
        except Exception:
            # Fallback to raw name if natural name lookup fails
            object_name = pickup_obj_name.replace("_", " ").title()

        self.config.task_config.referral_expressions["object_name"] = object_name
        self.config.task_config.referral_expressions_priority["object_name"] = [
            (1.0, 1.0, object_name)
        ]

        task: NavToObjTask = NavToObjTask(env, exp_config=self.config)
        # Store occupancy map reference in task for policy access
        task.occupancy_map = self._cached_thormap
        return task

    def _get_scene_objects(self, env: CPUMujocoEnv) -> list[MlSpacesObject]:
        """
        Get the list of candidate probjects in the scene for interactions.
        Filter by object types and prefer objects on the floor (not on furniture).
        """
        # Discover candidate nav_to objects
        om = env.object_managers[env.current_batch_index]
        candidates = om.get_objects_of_type(self.config.task_sampler_config.pickup_types)
        log.info(f"Found {len(candidates)} candidate nav objects in the scene")

        # Reject physically tiny/thin objects (pens, forks, ...) that are barely
        # visible in the shoulder cameras and give poor nav-learning signal. An
        # object is kept only if its SECOND-LARGEST full AABB extent (the smaller
        # of its two broadside dimensions) is at least ``min_object_size_m``.
        # ``aabb_size`` is the AABB HALF-extent, so full extent = 2 * aabb_size.
        min_size = float(getattr(self.config.task_sampler_config, "min_object_size_m", 0.0) or 0.0)
        if min_size > 0.0 and len(candidates) > 0:
            kept = []
            for obj in candidates:
                try:
                    full_extents = 2.0 * np.asarray(obj.aabb_size, dtype=float)
                except Exception:
                    kept.append(obj)  # can't measure -> don't drop
                    continue
                second_largest = float(np.sort(full_extents)[-2])
                if second_largest >= min_size:
                    kept.append(obj)
            n_dropped = len(candidates) - len(kept)
            if n_dropped > 0:
                log.info(
                    f"[nav size filter] dropped {n_dropped}/{len(candidates)} candidate(s) "
                    f"with second-largest extent < {min_size:.3f} m (too small/thin to be a good nav target)"
                )
            candidates = kept

        if not len(candidates) > 0:
            log.info("[WARN] No candidate nav objects found in the scene")
            # print all the top-level objects in the scene for debugging
            om = env.object_managers[env.current_batch_index]
            all_objects = MlSpacesObject.get_top_level_bodies(model=self.env.mj_model)
            for b in all_objects[:30]:
                name = self.env.mj_model.body(b).name
                pos = self.env.current_data.xpos[b]
                possible_types = om.get_possible_object_types(b)
                log.info(
                    f"  - #{b:02d} {name} (types={possible_types}) pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )

            # log.info(f"Scene objects (no candidates): {[obj.name for obj in all_objects]}")
            raise HouseInvalidForTask("No nav candidates found in the scene")

        return candidates

    def _snap_to_free(self, free_ds: np.ndarray, cell: tuple[int, int], max_r: int = 8):
        """Return the nearest free cell to ``cell`` in the downsampled grid.

        The requested cell (robot footprint / object-on-furniture) is often
        occupied; ring-search outward for the closest free cell so the geodesic
        planner has valid endpoints. Returns None if none within ``max_r``.
        """
        h, w = free_ds.shape
        r0, c0 = cell
        r0 = int(np.clip(r0, 0, h - 1))
        c0 = int(np.clip(c0, 0, w - 1))
        if free_ds[r0, c0]:
            return (r0, c0)
        for rad in range(1, max_r + 1):
            lo_r, hi_r = max(0, r0 - rad), min(h, r0 + rad + 1)
            lo_c, hi_c = max(0, c0 - rad), min(w, c0 + rad + 1)
            sub = free_ds[lo_r:hi_r, lo_c:hi_c]
            if sub.any():
                rr, cc = np.argwhere(sub).T
                d = np.abs(rr - (r0 - lo_r)) + np.abs(cc - (c0 - lo_c))
                k = int(np.argmin(d))
                return (lo_r + int(rr[k]), lo_c + int(cc[k]))
        return None

    def _detour_ratio(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
        min_ratio: float,
    ) -> float:
        """Geodesic-to-straight-line distance ratio between robot and target.

        Computes the geodesic (obstacle-avoiding) shortest-path distance between
        the robot and target on the cached per-house occupancy map and returns
        its ratio to the straight-line distance. A large ratio (>= ``min_ratio``)
        means a wall/large obstacle lies between them and the planner has to go
        around it -- i.e. the target is "on the other side of a wall". This is
        far more reliable than a line-of-sight test, which also trips on small
        furniture the planner merely skirts (negligible detour).

        Occupancy cells are True=free / False=occupied. Doorway paths are cleared
        in the map, so the geodesic routes through real doors. Returns ``0.0``
        (fail-open, i.e. "no detour") on any error or when no route exists, so
        the caller degrades to standard placement. ``min_ratio`` is used only for
        the optional debug log annotation.
        """
        thormap = self._cached_thormap
        if thormap is None:
            return 0.0
        try:
            from skimage.graph import route_through_array

            occ = np.asarray(thormap.occupancy, dtype=bool)  # True = free
            # Downsample to ~0.1 m cells for a fast geodesic (a free coarse cell
            # requires ALL underlying fine cells free, so walls stay solid).
            px_per_m = float(thormap.px_per_m)
            s = max(1, int(round(0.1 * px_per_m)))
            h, w = occ.shape
            hh, ww = (h // s) * s, (w // s) * s
            free_ds = occ[:hh, :ww].reshape(hh // s, s, ww // s, s).all(axis=(1, 3))
            if not free_ds.any():
                return 0.0
            cell_m = s / px_per_m

            rp = thormap.pos_m_to_px(np.asarray(robot_pos, dtype=float)[None, :])[0]
            gp = thormap.pos_m_to_px(np.asarray(target_pos, dtype=float)[None, :])[0]
            r_cell = self._snap_to_free(free_ds, (rp[0] // s, rp[1] // s))
            g_cell = self._snap_to_free(free_ds, (gp[0] // s, gp[1] // s))
            if r_cell is None or g_cell is None:
                return 0.0

            # Measure euclid between the SAME snapped free cells used as geodesic
            # endpoints (in meters), so geodesic >= euclid always holds and the
            # ratio purely reflects the go-around factor. Comparing against the
            # raw object center is wrong: objects rest inside occupied furniture,
            # so its cell is snapped to a free edge and the raw straight-line
            # distance over-counts, deflating the ratio below 1.0.
            euclid = float(
                np.linalg.norm(np.asarray(g_cell, float) - np.asarray(r_cell, float))
            ) * cell_m
            if euclid < 1e-3:
                return 0.0
            # cost 1 in free space, effectively impassable in occupied space.
            cost = np.where(free_ds, 1.0, 1e6).astype(np.float64)
            _, weight = route_through_array(
                cost, r_cell, g_cell, fully_connected=True, geometric=True
            )
            if weight >= 1e5:  # path had to cross a wall -> no real route
                import os as _os
                _dbg = _os.environ.get("MLSPACES_WB_DEBUG_FILE")
                if _dbg:
                    with open(_dbg, "a") as _fh:
                        _fh.write(f"NOROUTE weight={weight:.1f} euclid={euclid:.3f}\n")
                return 0.0
            geodesic_m = float(weight) * cell_m
            ratio = geodesic_m / euclid
            import os as _os
            _dbg = _os.environ.get("MLSPACES_WB_DEBUG_FILE")
            if _dbg:
                # true world distance (robot->object) for sanity vs snapped euclid
                true_d = float(np.linalg.norm(
                    np.asarray(target_pos, float)[:2] - np.asarray(robot_pos, float)[:2]))
                # LOS check: does the straight pixel line rp->gp cross any occupied cell?
                npts = max(2, int(np.hypot(gp[0]-rp[0], gp[1]-rp[1]) / s))
                rr = np.linspace(rp[0], gp[0], npts).astype(int)
                cc = np.linspace(rp[1], gp[1], npts).astype(int)
                rr = np.clip(rr, 0, occ.shape[0]-1); cc = np.clip(cc, 0, occ.shape[1]-1)
                los_blocked = int((~occ[rr, cc]).sum())  # occupied px on straight line
                with open(_dbg, "a") as _fh:
                    _fh.write(
                        f"euclid={euclid:.3f} geo={geodesic_m:.3f} ratio={ratio:.3f} "
                        f"true_d={true_d:.3f} los_blocked_px={los_blocked}/{npts} "
                        f"rob=({robot_pos[0]:.2f},{robot_pos[1]:.2f}) "
                        f"obj=({target_pos[0]:.2f},{target_pos[1]:.2f}) "
                        f"ok={ratio>=min_ratio}\n")
            return ratio
        except Exception as _e:  # noqa: BLE001 - never let the bias crash datagen
            import os as _os
            _dbg = _os.environ.get("MLSPACES_WB_DEBUG_FILE")
            if _dbg:
                with open(_dbg, "a") as _fh:
                    _fh.write(f"EXC {type(_e).__name__}: {_e}\n")
            return 0.0

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Sample a nav object, place robot using occupancy map, and return sampled params.

        Returns:
            dict with keys: pickup_obj_name, robot_base_pose
        Raise:
            RobotPlacementError if robot placement fails
        """
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        robot_view = env.current_robot.robot_view
        log.debug(f"Selected pickup object: {task_cfg.pickup_obj_name}")
        log.debug(f"[TASK SAMPLING] Trying to place robot near '{pickup_obj.name}'")

        # randomize pickup object texture
        if (
            self.texture_randomizer is not None
            and self.config.task_sampler_config.randomize_textures
        ):
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("robot_randomize_pickup_obj")
            self.texture_randomizer.randomize_object(pickup_obj)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("robot_randomize_pickup_obj")

        if isinstance(pickup_obj, MlSpacesObject):
            pickup_obj_pos = pickup_obj.position
        else:
            raise ValueError(f"Invalid pickup object type: {type(pickup_obj)}")

        # Positions of ALL same-type candidate instances. The navigation policy
        # and the success criterion target the NEAREST instance to the robot
        # (get_nav_object_priority), not necessarily ``pickup_obj``. The
        # wall-between predicate must therefore be evaluated against whichever
        # instance is nearest to each sampled robot pose, or a closer instance
        # in open space silently defeats the bias.
        wb_candidate_positions: list[np.ndarray] = []
        try:
            cand_names = getattr(task_cfg, "pickup_obj_candidates", None) or [pickup_obj.name]
            for _nm in cand_names:
                try:
                    wb_candidate_positions.append(np.asarray(om.get_object_by_name(_nm).position))
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            wb_candidate_positions = [np.asarray(pickup_obj_pos)]
        if not wb_candidate_positions:
            wb_candidate_positions = [np.asarray(pickup_obj_pos)]

        # Check if robot_base_pose is already set (e.g., from frozen_config)
        if task_cfg.robot_base_pose is not None:
            # Restore robot to saved pose instead of sampling
            from molmo_spaces.utils.pose import pos_quat_to_pose_mat

            log.info(f"Restoring robot from frozen_config: {task_cfg.robot_base_pose}")

            saved_pose = np.array(task_cfg.robot_base_pose)
            robot_view.base.pose = pos_quat_to_pose_mat(saved_pose[:3], saved_pose[3:])

            final_pos = robot_view.base.pose[:3, 3]
            distance_to_obj = np.linalg.norm(final_pos[:2] - pickup_obj_pos[:2])
            log.info("[OK] Robot restored from config")
            log.info(
                f"Final robot position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})"
            )
            log.info(f"Distance to object: {distance_to_obj:.3f}m")
        else:
            # Sample a new robot position
            # Log placement parameters
            sampling_radius_range = self.config.task_sampler_config.base_pose_sampling_radius_range
            robot_safety_radius = self.config.task_sampler_config.robot_safety_radius
            initial_robot_z = self.config.task_sampler_config.robot_object_z_offset
            max_robot_placement_attempts = (
                self.config.task_sampler_config.max_robot_placement_attempts
            )
            face_target = self.config.task_sampler_config.face_target

            log.info(
                f"Attempting to place robot near '{pickup_obj.name}' in radius range {sampling_radius_range[0]:.3f}m - {sampling_radius_range[1]:.3f}m"
            )

            # place robot near pickup object
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("robot_place_near_pickup_obj")

            require_wall_between = bool(
                getattr(self.config.task_sampler_config, "require_wall_between", False)
            )
            if not require_wall_between:
                robot_placed = env.place_robot_near(
                    robot_view=robot_view,
                    target=pickup_obj,
                    max_tries=max_robot_placement_attempts,
                    sampling_radius_range=sampling_radius_range,
                    robot_safety_radius=robot_safety_radius,
                    preserve_z=initial_robot_z,
                    face_target=face_target,
                    # check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
                    # visibility_resolver=self.get_visibility_resolver(env),
                    # excluded_positions=self.used_robot_positions[pickup_obj.name],
                )
            else:
                # "Wall-between" bias: reject-sample placements until the robot
                # must route AROUND a wall to reach the target (geodesic path >>
                # straight line), i.e. the target sits on the other side of a
                # wall. Each attempt re-runs the standard placement (its own
                # collision + radius sampling), then tests the geodesic detour
                # ratio on the cached occupancy map. Fall back to the last valid
                # placement if the predicate can't be met (yield-preserving).
                cfg = self.config.task_sampler_config
                wb_max = int(getattr(cfg, "wall_between_max_attempts", 40))
                wb_min_ratio = float(getattr(cfg, "wall_between_min_geodesic_ratio", 1.4))
                wb_max_ratio = float(getattr(cfg, "wall_between_max_geodesic_ratio", 2.5))
                wb_fallback = bool(getattr(cfg, "wall_between_fallback", True))
                # Constrain the spawn distance for wall-between placements. The
                # default nav radius (up to ~20 m) produces go-arounds so long the
                # oracle nav times out / fails; a tighter band keeps the detour
                # genuine but achievable, which both raises the accepted-demo yield
                # and cuts per-episode time.
                wb_radius = getattr(cfg, "wall_between_sampling_radius_range", None)
                wb_sampling_radius_range = (
                    tuple(wb_radius) if wb_radius else sampling_radius_range
                )
                robot_placed = False
                wall_between_found = False
                for wb_attempt in range(wb_max):
                    placed = env.place_robot_near(
                        robot_view=robot_view,
                        target=pickup_obj,
                        max_tries=max_robot_placement_attempts,
                        sampling_radius_range=wb_sampling_radius_range,
                        robot_safety_radius=robot_safety_radius,
                        preserve_z=initial_robot_z,
                        face_target=face_target,
                    )
                    if not placed:
                        continue
                    robot_placed = True  # at least one collision-free pose exists
                    robot_xy = robot_view.base.pose[:3, 3]
                    # Evaluate the predicate against the NEAREST candidate -- the
                    # instance the policy will actually navigate to. Require a
                    # genuine but ACHIEVABLE go-around (ratio in
                    # [min_ratio, max_ratio]); extreme detours to a far instance
                    # make the oracle nav time out and yield nothing, so cap them.
                    rp2 = np.asarray(robot_xy, float)[:2]
                    nearest = min(
                        wb_candidate_positions,
                        key=lambda p: float(np.linalg.norm(p[:2] - rp2)),
                    )
                    ratio = self._detour_ratio(robot_xy, nearest, wb_min_ratio)
                    if wb_min_ratio <= ratio <= wb_max_ratio:
                        wall_between_found = True
                        log.info(
                            f"[wall-between] satisfied after {wb_attempt + 1} attempt(s) "
                            f"for target '{pickup_obj.name}' (nearest-instance ratio={ratio:.2f})"
                        )
                        break
                if robot_placed and not wall_between_found:
                    if wb_fallback:
                        log.info(
                            f"[wall-between] no around-a-wall placement in {wb_max} attempts for "
                            f"'{pickup_obj.name}'; falling back to direct placement (yield-preserving)"
                        )
                    else:
                        robot_placed = False

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("robot_place_near_pickup_obj")

            if not robot_placed:
                log.info(f"[FAIL] Failed to place robot near '{pickup_obj.name}'")
                raise RobotPlacementError(f"Failed to place robot near object: {pickup_obj.name}")

            # Get final robot pose for return data
            task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
            final_pos = robot_view.base.pose[:3, 3]
            import os as _os
            _dbg = _os.environ.get("MLSPACES_WB_DEBUG_FILE")
            if _dbg and require_wall_between:
                with open(_dbg, "a") as _fh:
                    _fh.write(
                        f"FINAL wb_found={wall_between_found} "
                        f"pose=({final_pos[0]:.2f},{final_pos[1]:.2f}) "
                        f"obj=({pickup_obj_pos[0]:.2f},{pickup_obj_pos[1]:.2f})\n")
            log.info("[OK] Successfully placed robot")
            log.info(
                f"Final robot position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})"
            )
            log.info(
                f"Object position: ({pickup_obj_pos[0]:.3f}, {pickup_obj_pos[1]:.3f}, {pickup_obj_pos[2]:.3f})"
            )
