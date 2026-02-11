import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.object_manager import Context, ObjectManager
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.tasks.task_sampler_errors import (
    HouseInvalidForTask,
    ObjectPlacementError,
    RobotPlacementError,
)
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask, load_grasps_for_object
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.mj_model_and_data_utils import body_base_pos
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom, place_object_near
from molmo_spaces.utils.object_metadata import ObjectMeta
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_7d
from molmo_spaces.utils.synset_utils import get_valid_receptacle_uids

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_and_place_configs import PickAndPlaceDataGenConfig


log = logging.getLogger(__name__)


# Maximum difference in bottom Z between pickup object and place target
# to consider them on the same surface
MAX_BOTTOM_Z_DIFFERENCE = 0.05  # 5cm

# Cached valid receptacle UIDs for efficiency
_VALID_RECEPTACLE_CACHE: dict[str, dict] | None = None


def _get_cached_valid_receptacles() -> dict[str, dict]:
    """Get cached valid receptacle UIDs filtered by synset rules."""
    global _VALID_RECEPTACLE_CACHE
    if _VALID_RECEPTACLE_CACHE is None:
        _VALID_RECEPTACLE_CACHE = get_valid_receptacle_uids()
    return _VALID_RECEPTACLE_CACHE


class MetadataAdder:
    def __init__(self, name_to_meta):
        import threading

        self.pending = True
        self.semaphore = threading.Semaphore()
        self.name_to_meta = name_to_meta

    def add_meta(self, metadata):
        if self.pending:
            self.semaphore.acquire()
            try:
                if self.pending:
                    for name, meta in self.name_to_meta.items():
                        if name not in metadata["objects"]:
                            metadata["objects"][name] = meta

                    self.pending = False
            finally:
                self.semaphore.release()


class PickAndPlaceTaskSampler(PickTaskSampler):
    def __init__(self, config: "PickAndPlaceDataGenConfig") -> None:
        self.place_receptacle_name = None
        super().__init__(config)
        self.config: PickAndPlaceDataGenConfig
        self._receptacle_cache = {}
        self._metadata_adder = None
        # Multiple receptacles added to scene for fallback
        self._receptacle_names: list[str] = []
        self._receptacle_uids: list[str] = []
        # Index of current receptacle being used
        self._current_receptacle_index: int = 0
        # Counter for episodes with current receptacle (for auto-advance)
        self._episodes_with_current_receptacle: int = 0
        self._receptacle_multiplier = 1
        self._receptacle_staging_poses = {}

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Add task-specific objects to scene."""
        super().add_auxiliary_objects(spec)
        self._add_receptacles_to_scene(spec)

    def _material_callback(self, object_spec: mujoco.MjsBody):
        pass

    def _add_receptacles_to_scene(self, spec: MjSpec) -> None:
        """Add receptacle objects to scene for place-in-receptacle tasks."""

        max_size = np.array([0.5, 0.5, 0.15])
        min_size = np.array([0.17, 0.17, -1])

        def valid_receptacle(anno):
            # We select smallish receptacles (pickupable), so that
            # they can be placed on the work surface/bench and
            # prefer wide (vs tall) receptacles
            xyz = [anno["boundingBox"][x] for x in "xyz"]
            return (
                anno["receptacle"]
                and anno["primaryProperty"] == "CanPickup"
                and max_size[2] >= xyz[2]
                and max_size[0] >= xyz[0] > min_size[0]  # > xyz[2]
                and max_size[1] >= xyz[1] > min_size[1]  # > xyz[2]
            )

        task_sampler_config = self.config.task_sampler_config

        # Use synset-based filtering if place_receptacle_types is None or empty
        if not task_sampler_config.place_receptacle_types:
            # Use synset-based receptacle filtering
            cache_key = "synset_receptacles"
            if cache_key not in self._receptacle_cache:
                all_valid = _get_cached_valid_receptacles()
                valid_uids = sorted(
                    [uid for uid, anno in all_valid.items() if valid_receptacle(anno)]
                )
                self._receptacle_cache[cache_key] = {uid: all_valid[uid] for uid in valid_uids}
            valid_uids = sorted(self._receptacle_cache[cache_key].keys())
        else:
            # Legacy: use explicit receptacle type list
            recep_types = list(task_sampler_config.place_receptacle_types)
            np.random.shuffle(recep_types)

            cache_key: str = None
            for it in range(len(recep_types)):
                cache_key = recep_types[it].lower()

                if cache_key not in self._receptacle_cache:
                    uid_to_anno = ObjectManager.uid_to_annotation_for_type(cache_key)
                    valid_uids = sorted(
                        [uid for uid, anno in uid_to_anno.items() if valid_receptacle(anno)]
                    )

                    valid_uids = ObjectManager.prefilter_with_clip(cache_key, valid_uids)

                    self._receptacle_cache[cache_key] = {
                        uid: uid_to_anno[uid] for uid in valid_uids
                    }

                    if len(valid_uids):
                        break
                else:
                    valid_uids = sorted(self._receptacle_cache[cache_key].keys())
                    if len(valid_uids):
                        break
            else:
                valid_uids = []

        if len(valid_uids) == 0:
            raise ValueError("No valid receptacle assets found")

        # Add multiple receptacles for fallback (default 2, configurable)
        num_receptacles = getattr(task_sampler_config, "num_place_receptacles", 2)
        num_receptacles = min(num_receptacles, len(valid_uids))

        # Sample N unique receptacles
        selected_uids = list(np.random.choice(valid_uids, size=num_receptacles, replace=False))

        multiplier = self._receptacle_multiplier

        # Reset receptacle tracking
        self._receptacle_names = []
        self._receptacle_uids = []
        self._current_receptacle_index = 0
        name_to_meta = {}

        # Add a static floor below the off-screen receptacle staging area
        # 1 m per receptacle and per multiplier, in geom box, size is half side,
        # so box has size num_receptacles x multiplier x 1
        staging_size = np.array([num_receptacles, multiplier, 1]) / 2

        # place it somewhere way below the scene (min_z = 0)
        staging_center = np.array([5, 5, 15])

        # start placing on the top xy corner of the top face, with a 0.5m margin from the edge
        staging_start = staging_center + np.array(
            [0.5 - staging_size[0], 0.5 - staging_size[1], staging_size[2]]
        )

        mocap_body = spec.worldbody.add_body(
            name="receptacle_staging_floor",
            mocap=True,
            pos=staging_center,
        )

        mocap_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=staging_size,
            # TODO make sure this is the same used for colliders in house gen
            contype=8,
            conaffinity=15,
            group=4,
        )

        self._receptacle_staging_poses = {}

        for i, uid in enumerate(selected_uids):
            receptacle_xml = install_uid(uid)
            for j in range(multiplier):
                receptacle_spec = MjSpec.from_file(str(receptacle_xml))
                if len(receptacle_spec.worldbody.bodies) != 1:
                    log.warning(
                        f"{receptacle_xml} has {len(receptacle_spec.worldbody.bodies)} bodies, expected 1."
                    )
                receptacle_obj: mujoco.MjsBody = receptacle_spec.worldbody.bodies[0]
                self._material_callback(receptacle_obj)

                if not receptacle_obj.first_joint():
                    receptacle_obj.add_joint(
                        name=f"{uid}_copy{j}_jntfree",
                        type=mujoco.mjtJoint.mjJNT_FREE,
                        damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
                    )

                # make the bottom of the receptacle be at the same height as the top
                # of the staging box + a small offset to be solved by gravity
                z_shift = self._receptacle_cache[cache_key][uid]["boundingBox"]["z"] / 2 + 0.01

                position = staging_start + np.array([i, j, z_shift])
                # orig y is now z, orig -z is now y? oh, well
                quat = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)

                attach_frame = spec.worldbody.add_frame(
                    pos=position,
                    quat=quat,
                )
                namespace = f"{task_sampler_config.place_receptacle_namespace}{i}_{j}/"
                attach_frame.attach_body(receptacle_obj, namespace, "")

                self._receptacle_names.append(receptacle_obj.name)
                self._receptacle_uids.append(uid)

                self._receptacle_staging_poses[receptacle_obj.name] = np.concatenate(
                    (position, quat)
                )

                xml_path_rel = receptacle_xml.relative_to(ASSETS_DIR)
                self.config.task_config.added_objects[receptacle_obj.name] = xml_path_rel

                uid_anno = self._receptacle_cache[cache_key][uid]
                name_to_meta[receptacle_obj.name] = {
                    "asset_id": uid,
                    "category": uid_anno["category"],
                    "object_enum": "temp_object",
                    "is_static": False,  # it always has a free joint
                    "boundingBox": uid_anno.get("boundingBox", {}),
                }

        self.place_receptacle_name = self._receptacle_names[0]
        log.info(
            f"Added {num_receptacles} (x {multiplier}) receptacles to scene: {self._receptacle_uids}"
        )

        self._metadata_adder = MetadataAdder(name_to_meta)

    def resolve_visibility_object(self, env: CPUMujocoEnv, key: str) -> list[str]:
        """Resolve special visibility object keys.

        Handles:
        - __task_objects__: pickup object and place receptacle
        - __gripper__: Robot gripper (via base class)
        """
        resolved_objects = super().resolve_visibility_object(env, key)
        if key == "__task_objects__" and self.place_receptacle_name is not None:
            resolved_objects.append(self.place_receptacle_name)
        return resolved_objects

    def advance_to_next_receptacle(self, env: CPUMujocoEnv) -> bool:
        """Advance to the next receptacle in the preloaded set.

        Call this when the current receptacle causes failures (e.g., IK failures
        because the receptacle is too high). Returns True if there's another
        receptacle to try, False if all have been exhausted.

        Returns:
            True if successfully advanced to next receptacle, False if no more available.
        """
        multiplier = self._receptacle_multiplier

        # Kept in case we want to allow sampling multiple receptacles for each pickup object
        # om = env.object_managers[env.current_batch_index]
        #
        # for name in self.active_receptacle_names:
        #     obj = create_mlspaces_body(om.data, name)
        #     obj.position = np.array(self._receptacle_staging_poses[name][:3])
        #     obj.quat = np.array(self._receptacle_staging_poses[name][3:])
        #
        # mujoco.mj_fwdPosition(om.model, om.data)

        if self._current_receptacle_index + multiplier < len(self._receptacle_names):
            self._current_receptacle_index += multiplier
            self.place_receptacle_name = self._receptacle_names[self._current_receptacle_index]
            self._episodes_with_current_receptacle = 0  # Reset counter
            log.info(
                f"Advanced to receptacle {self._current_receptacle_index // multiplier + 1}/{len(self._receptacle_names) // multiplier}: "
                f"{self._receptacle_uids[self._current_receptacle_index]}"
            )
            return True
        else:
            log.info("No more receptacles available to try")
            return False

    def reset_receptacle_index(self) -> None:
        """Reset to the first receptacle.

        Call this when starting a new task within the same scene.
        """
        self._current_receptacle_index = 0
        self._episodes_with_current_receptacle = 0
        if self._receptacle_names:
            self.place_receptacle_name = self._receptacle_names[0]

    @property
    def current_receptacle_uid(self) -> str | None:
        """Get the UID of the currently active receptacle."""
        if self._receptacle_uids and self._current_receptacle_index < len(self._receptacle_uids):
            return self._receptacle_uids[self._current_receptacle_index]
        return None

    @property
    def has_more_receptacles(self) -> bool:
        """Check if there are more receptacles to try."""
        multiplier = self._receptacle_multiplier
        return self._current_receptacle_index + multiplier < len(self._receptacle_names)

    @property
    def active_receptacle_names(self):
        multiplier = self._receptacle_multiplier
        return self._receptacle_names[
            self._current_receptacle_index : self._current_receptacle_index + multiplier
        ]

    def _get_place_target_candidates(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        supporting_geom_id: int,
    ) -> list[str]:
        """Get candidate place target object names.

        Parent class returns pre-added receptacles.
        Override in child classes for different strategies (e.g., bench objects).

        Args:
            env: The environment.
            pickup_obj_name: Name of the pickup object.
            supporting_geom_id: Geom ID of the surface supporting the pickup object.

        Returns:
            List of candidate place target object names.
        """
        return self._receptacle_names

    def _prepare_place_target(
        self,
        env: CPUMujocoEnv,
        place_target_name: str,
        pickup_obj_name: str,
        pickup_obj_pos: np.ndarray,
        supporting_geom_id: int,
    ) -> bool:
        """Position the place target near the pickup object.

        Parent class moves receptacle near pickup object using place_object_near.
        Override in child classes if no positioning needed (e.g., objects already placed).

        Args:
            env: The environment.
            place_target_name: Name of the place target object.
            pickup_obj_pos: Position of the pickup object.
            supporting_geom_id: Geom ID of the surface supporting the pickup object.

        Returns:
            True if successful, False otherwise.
        """
        task_sampler_config = self.config.task_sampler_config

        om = env.object_managers[env.current_batch_index]
        data = env.current_data

        # Place ALL receptacles near the pickup object
        for receptacle_name in self.active_receptacle_names:
            if not self._filter_place_target(env, pickup_obj_name, receptacle_name):
                log.info(f"Place receptacle {receptacle_name} fails filter size")
                return False

            receptacle_id = om.get_object_body_id(receptacle_name)

            try:
                place_object_near(
                    data=env.current_data,
                    object_id=receptacle_id,
                    placement_point=pickup_obj_pos,
                    min_dist=task_sampler_config.min_object_to_receptacle_dist,
                    max_dist=task_sampler_config.max_object_to_receptacle_dist,
                    max_tries=task_sampler_config.max_place_receptacle_sampling_attempts,
                    max_dist_to_reference=task_sampler_config.max_robot_to_place_receptacle_dist,
                    supporting_geom_id=supporting_geom_id,
                    z_eps=0.003,
                )
            except ObjectPlacementError:
                log.info(f"Failed to place receptacle {receptacle_name} near pickup object")
                return False

            # Filter to objects on the same surface (similar bottom Z)
            r_obj = om.get_object(receptacle_name)
            r_base_pos = body_base_pos(data, r_obj.body_id)

            if abs(r_base_pos[2] - pickup_obj_pos[2]) > MAX_BOTTOM_Z_DIFFERENCE:
                raise ValueError(
                    f"Failed to place receptacle {receptacle_name} at same height as pickup object"
                )

        return True

    def _filter_place_target(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        place_target_name: str,
    ) -> bool:
        """Check if pickup object fits in/on place target.

        Parent class ensures pickup object is smaller than receptacle.
        Override in child classes if no size constraint needed.

        Args:
            env: The environment.
            pickup_obj_name: Name of the pickup object.
            place_target_name: Name of the place target object.

        Returns:
            True if valid, False if should be rejected.
        """
        pickup_asset_id = (
            env.current_scene_metadata["objects"].get(pickup_obj_name, {}).get("asset_id")
        )
        if pickup_asset_id is None:
            log.info(f"Failed to get asset_id for {pickup_obj_name}")
            return False

        pickup_anno = ObjectMeta.annotation(pickup_asset_id)
        if pickup_anno is None:
            log.info(f"Failed to get annotation for {pickup_obj_name}")
            return False

        place_target_meta = env.current_scene_metadata["objects"].get(place_target_name)
        if place_target_meta is None or "boundingBox" not in place_target_meta:
            log.info(f"Failed to get bounding box for {place_target_name}")
            return False

        max_diag = float(np.linalg.norm([place_target_meta["boundingBox"][x] for x in "xyz"]))
        pickup_diag = float(np.linalg.norm([pickup_anno["boundingBox"][x] for x in "xyz"]))

        if pickup_diag > max_diag:
            log.info(
                f"Excluded pickup object {pickup_obj_name} with diag {pickup_diag:.3f} "
                f"larger than place target's {max_diag:.3f}"
            )
            return False

        return True

    def _sample_task(self, env: CPUMujocoEnv) -> PickAndPlaceTask:
        """Sample a pick-and-place task configuration and create the task."""
        # Set current batch index to 0 (most common case for single-batch environments)
        # TODO(rose) at some point: handle multi-batch environments properly
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        # Auto-advance receptacle after N episodes (if configured)
        episodes_per_receptacle = getattr(
            self.config.task_sampler_config, "episodes_per_receptacle", 0
        )
        if episodes_per_receptacle > 0:
            self._episodes_with_current_receptacle += 1
            if self._episodes_with_current_receptacle > episodes_per_receptacle:
                if self.has_more_receptacles:
                    self.advance_to_next_receptacle(env)
                else:
                    # Wrap around to first receptacle
                    self._current_receptacle_index = 0
                    if self._receptacle_names:
                        self.place_receptacle_name = self._receptacle_names[0]
                    self._episodes_with_current_receptacle = 1
                    log.info("Wrapped around to first receptacle")

        if self._metadata_adder is not None:
            self._metadata_adder.add_meta(env.current_scene_metadata)

        om = env.object_managers[env.current_batch_index]

        keep_task_cfg = self.config.task_config.pickup_obj_name is not None

        filtered_receptacle_expression_priority: list[tuple[float, float, str]] = None
        receptacle_expression_priority: list[tuple[float, float, str]] = None
        filtered_expression_priority: list[tuple[float, float, str]] = None
        expression_priority: list[tuple[float, float, str]] = None

        sample_success = False
        max_attempts = len(self.candidate_objects)
        attempts = 0
        while not sample_success and len(self.candidate_objects) > 0 and attempts < max_attempts:
            attempts += 1

            if not keep_task_cfg:
                self.config.task_config.pickup_obj_name = None

            # Sample pickup object
            if self.config.task_config.pickup_obj_name is None:
                object_index = self._task_counter % len(self.candidate_objects)
                self.config.task_config.pickup_obj_name = self.candidate_objects[object_index].name
                log.info(
                    f"Attempting object {self.config.task_config.pickup_obj_name} {object_index}/{len(self.candidate_objects)}"
                )
            else:
                log.info(
                    f"Attempting object {self.config.task_config.pickup_obj_name} of {len(self.candidate_objects)}"
                )
            self._task_counter += 1  # update counter, so we don't re-try same object

            pickup_obj_name = self.config.task_config.pickup_obj_name
            pickup_obj_id = env.current_model.body(pickup_obj_name).id
            pickup_obj_pos = body_base_pos(env.current_data, pickup_obj_id)

            supporting_geom_id = get_supporting_geom(env.current_data, pickup_obj_id)
            if supporting_geom_id is None:
                log.info(f"Failed to get supporting geom_id for {pickup_obj_name}")
                # Remove - no supporting surface won't change on retry
                self._remove_candidate_object(pickup_obj_name)
                continue

            # Get place target candidates (overridable)
            place_candidates = self._get_place_target_candidates(
                env, pickup_obj_name, supporting_geom_id
            )

            if not place_candidates:
                log.info(f"No place target candidates for {pickup_obj_name}")
                self._remove_candidate_object(pickup_obj_name)
                continue

            # Prepare/position place target (overridable)
            try:
                if not self._prepare_place_target(
                    env,
                    self.place_receptacle_name,
                    pickup_obj_name,
                    pickup_obj_pos,
                    supporting_geom_id,
                ):
                    log.info(f"No valid place target found for {pickup_obj_name}")
                    # Keeping the object for possible future receptacles
                    continue
            except ValueError:
                log.exception(f"Removing {pickup_obj_name}.")
                self._remove_candidate_object(pickup_obj_name)
                continue

            # Update config with selected place target
            self.config.task_config.place_receptacle_name = self.place_receptacle_name
            self.config.task_config.place_target_name = self.place_receptacle_name

            # Setup cameras initially so they are available for visibility checks during placement
            self.setup_cameras(env, deterministic_only=True)

            # Record place receptacle start pose
            receptacle_obj = om.get_object_by_name(self.place_receptacle_name)
            self.config.task_config.place_receptacle_start_pose = pose_mat_to_7d(
                receptacle_obj.pose
            ).tolist()

            #  place robot accordingly
            try:
                self._sample_and_place_robot(env)
            except RobotPlacementError as e:
                log.info(f"Robot placement failed for {pickup_obj_name}: {e}")
                # Report failure for this asset (may lead to dynamic blacklisting)
                asset_uid = self.get_asset_uid_from_object(env, pickup_obj_name)
                if asset_uid:
                    self.report_asset_failure(asset_uid, f"robot placement failed: {e}")
                # Remove this object from candidates - don't retry failed placements
                self._remove_candidate_object(pickup_obj_name)
                continue

            # Ensure robot is in final position before camera setup
            mujoco.mj_forward(env.current_model, env.current_data)

            # Check grasp feasibility before proceeding
            pickup_obj = om.get_object_by_name(pickup_obj_name)
            asset_uid = self.get_asset_uid_from_object(env, pickup_obj_name)
            if asset_uid:
                try:
                    _gripper, cached_grasps = load_grasps_for_object(asset_uid, 512)
                    if len(cached_grasps) > 0:
                        object_pose = pos_quat_to_pose_mat(pickup_obj.position, pickup_obj.quat)
                        grasp_poses_world = object_pose @ cached_grasps
                        noncolliding_mask = get_noncolliding_grasp_mask(
                            env.current_model, env.current_data, grasp_poses_world, 64
                        )
                        n_feasible = int(np.sum(noncolliding_mask))
                        if n_feasible == 0:
                            log.info(
                                f"No feasible grasps for {pickup_obj_name} (uid={asset_uid}): "
                                f"0/{len(grasp_poses_world)} non-colliding"
                            )
                            self.report_grasp_failure(pickup_obj_name)
                            continue
                except ValueError:
                    log.info(f"No grasps found for {pickup_obj_name} (uid={asset_uid})")
                    self.report_grasp_failure(pickup_obj_name)
                    continue

            # Note: choosing referral expressions via sampling
            context_objects = om.get_context_objects(
                pickup_obj_name, Context.BENCH, bench_geom_ids=[supporting_geom_id]
            )

            context_names = {obj.name for obj in context_objects}

            # Ensure the first (target receptacle) is in context
            if self.place_receptacle_name not in context_names:
                context_objects.append(om.get_object(self.place_receptacle_name))

            if self._receptacle_multiplier > 1:
                # Ensure no other, e.g. when multiple receptacles are in context
                remove_from_context = {
                    recep_name
                    for recep_name in self.active_receptacle_names
                    if recep_name != self.place_receptacle_name
                }

                context_objects = [
                    obj for obj in context_objects if obj.name not in remove_from_context
                ]

            if pickup_obj_name not in context_names:
                context_objects.append(om.get_object(pickup_obj_name))

            try:
                expression_priority = om.referral_expression_priority(
                    pickup_obj_name, context_objects
                )
                filtered_expression_priority = om.thresholded_expression_priority(
                    expression_priority
                )
                # If threshold filtering removed all expressions, use unfiltered
                if len(filtered_expression_priority) == 0:
                    log.info(
                        f"No filtered expression priorities for {pickup_obj_name}, "
                        f"using unfiltered ({len(expression_priority)} expressions)"
                    )
                    filtered_expression_priority = expression_priority
            except NameError:
                expression_priority = [(1.0, 1.0, om.fallback_expression(pickup_obj_name))]
                filtered_expression_priority = expression_priority

            # If still empty, use fallback
            if len(filtered_expression_priority) == 0:
                log.info(
                    f"No expression priorities for pickup object {pickup_obj_name}, using fallback"
                )
                expression_priority = [(1.0, 1.0, om.fallback_expression(pickup_obj_name))]
                filtered_expression_priority = expression_priority

            try:
                receptacle_expression_priority = om.referral_expression_priority(
                    self.place_receptacle_name, context_objects
                )
                filtered_receptacle_expression_priority = om.thresholded_expression_priority(
                    receptacle_expression_priority
                )
            except NameError:
                receptacle_expression_priority = [
                    (
                        1.0,
                        1.0,
                        om.fallback_expression(self.place_receptacle_name),
                    )
                ]
                filtered_receptacle_expression_priority = receptacle_expression_priority

            # if len(filtered_receptacle_expression_priority) == 0:
            #     log.info(f"No filtered receptacle expression priorities for {pickup_obj_name}")
            #     # Remove - expression priority won't change on retry
            #     self._remove_candidate_object(pickup_obj_name)
            #     continue

            # Setup cameras after pickup object and robot placement
            # This allows cameras to use task-specific info (pickup object, workspace center)
            self.setup_cameras(env)

            # # Remove successfully used pickup object to avoid re-using it in this house
            # self._remove_candidate_object(pickup_obj_name)

            sample_success = True
            break

        if not sample_success:
            raise HouseInvalidForTask(
                f"Unable to sample a valid task after {attempts} attempts, {len(self.candidate_objects)} candidates remaining"
            )

        self.config.task_config.referral_expressions["pickup_name"] = om.sample_expression(
            filtered_expression_priority
        )
        self.config.task_config.referral_expressions_priority["pickup_name"] = expression_priority

        self.config.task_config.referral_expressions["place_name"] = om.sample_expression(
            filtered_receptacle_expression_priority
        )
        self.config.task_config.referral_expressions_priority["place_name"] = (
            receptacle_expression_priority
        )

        task = PickAndPlaceTask(env, self.config)
        return task
