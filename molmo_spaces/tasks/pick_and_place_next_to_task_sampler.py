import logging
from typing import TYPE_CHECKING

import numpy as np
from mujoco import MjSpec

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.object_manager import Context
from molmo_spaces.tasks.pick_and_place_next_to_task import PickAndPlaceNextToTask
from molmo_spaces.tasks.pick_and_place_task_sampler import (
    MAX_BOTTOM_Z_DIFFERENCE,
    PickAndPlaceTaskSampler,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_and_place_next_to_configs import (
        PickAndPlaceNextToDataGenConfig,
    )

log = logging.getLogger(__name__)


class PickAndPlaceNextToTaskSampler(PickAndPlaceTaskSampler):
    def __init__(self, config: "PickAndPlaceNextToDataGenConfig") -> None:
        super().__init__(config)
        self.config: PickAndPlaceNextToDataGenConfig

    # Note: Commenting out all this to keep default behavior
    # with added receptacles (this didn't work, but not sure if we want
    # to try something along these lines in the future)

    def _add_receptacles_to_scene(self, spec: MjSpec) -> None:
        """No-op: we use existing scene objects as place targets."""
        pass

    def _get_place_target_candidates(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        supporting_geom_id: int,
    ) -> list[str]:
        """Return objects on the same bench, excluding the pickup object.

        Filters to objects whose type is in place_receptacle_types config.
        Filters to objects on the same surface (similar bottom Z).
        Uses synset-based comparison to prefer objects of different types.
        If an object is labeled by a hypernym in context, objects that are
        hyponyms of that hypernym are considered the same type.
        """
        om = env.object_managers[env.current_batch_index]
        data = env.current_data

        # Get objects on the same bench/supporting surface
        context_objects = om.get_context_objects(
            pickup_obj_name, Context.BENCH, bench_geom_ids=[supporting_geom_id]
        )

        if not context_objects:
            self._receptacle_names = []
            return self._receptacle_names

        # Get pickup object's bottom Z
        pickup_obj = om.get_object_by_name(pickup_obj_name)
        pickup_center, pickup_size = body_aabb(data.model, data, pickup_obj.body_id)
        pickup_bottom_z = pickup_center[2] - pickup_size[2] / 2

        # Filter to objects on the same surface (similar bottom Z)
        same_surface_objects = []
        for obj in context_objects:
            obj_center, obj_size = body_aabb(data.model, data, obj.body_id)
            obj_bottom_z = obj_center[2] - obj_size[2] / 2
            if abs(obj_bottom_z - pickup_bottom_z) <= MAX_BOTTOM_Z_DIFFERENCE:
                same_surface_objects.append(obj)
        context_objects = same_surface_objects

        if not context_objects:
            self._receptacle_names = []
            return self._receptacle_names

        # Filter to objects matching place_receptacle_types
        place_receptacle_types = self.config.task_sampler_config.place_receptacle_types
        if place_receptacle_types:
            place_types_set = set(t.lower() for t in place_receptacle_types)
            filtered_context_objects = []
            for obj in context_objects:
                obj_types = om.get_possible_object_types(obj.name)
                if any(t.lower() in place_types_set for t in obj_types):
                    filtered_context_objects.append(obj)
            context_objects = filtered_context_objects

        if not context_objects:
            self._receptacle_names = []
            return self._receptacle_names

        # Get context synsets for all bench objects (handles hypernym/hyponym relationships)
        context_synsets = om.get_context_synsets(context_objects)

        # Get pickup object's required hypernyms within the bench context
        pickup_hypernyms = set(om.get_object_hypernyms(pickup_obj_name, context_synsets))
        pickup_synset = om.most_concrete_synset(pickup_hypernyms)

        different_synset = []
        same_synset = []

        for obj in context_objects:
            # Skip the pickup object itself
            if obj.name == pickup_obj_name:
                continue

            # Get this object's hypernyms within the bench context
            obj_hypernyms = set(om.get_object_hypernyms(obj.name, context_synsets))
            if obj_hypernyms:
                obj_synset = om.most_concrete_synset(obj_hypernyms)
            else:
                continue

            # If synsets match, objects could be referred to by the same name
            if pickup_synset == obj_synset:
                same_synset.append(obj.name)
            else:
                different_synset.append(obj.name)

        # Shuffle each group for variety
        np.random.shuffle(different_synset)
        np.random.shuffle(same_synset)

        # Combine candidates, preferring different synset (unambiguous)
        all_candidates = different_synset + same_synset

        # Rotate the list based on task counter to vary which candidate is tried first
        # This ensures different pairs are tried across episodes
        if all_candidates and self._task_counter is not None:
            rotation = self._task_counter % len(all_candidates)
            all_candidates = all_candidates[rotation:] + all_candidates[:rotation]

        self._receptacle_names = all_candidates
        return self._receptacle_names

    def _prepare_place_target(
        self,
        env: CPUMujocoEnv,
        place_target_name: str,
        pickup_obj_name: str,
        pickup_obj_pos: np.ndarray,
        supporting_geom_id: int,
    ) -> bool:
        """Check that place target is at a reasonable distance from pickup object.

        Objects are already placed in scene. We reject candidates that are:
        - Too close: already within success range (task would be trivial)
        - Too far: beyond max_object_to_receptacle_dist (IK would likely fail)
        """
        om = env.object_managers[env.current_batch_index]
        data = env.current_data
        task_sampler_config = self.config.task_sampler_config

        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)

        max_dist = task_sampler_config.max_object_to_receptacle_dist
        min_dist = task_sampler_config.min_object_to_receptacle_dist

        # Try each candidate place target
        place_target_found = False
        for place_target_name in self._receptacle_names:
            place_target = om.get_object_by_name(place_target_name)

            # Get AABBs for distance calculation
            pickup_center, pickup_size = body_aabb(data.model, data, pickup_obj.body_id)
            target_center, target_size = body_aabb(data.model, data, place_target.body_id)

            # Compute center-to-center distance in XY plane
            center_to_center_dist = np.linalg.norm(target_center[:2] - pickup_center[:2])

            # Reject if too far (beyond max_object_to_receptacle_dist)
            if center_to_center_dist > max_dist:
                log.info(
                    f"Rejecting {place_target_name}: too far from pickup object "
                    f"(center distance={center_to_center_dist:.3f}m > max={max_dist:.3f}m)"
                )
                continue

            # Reject if too close (use min_object_to_receptacle_dist from config)
            if center_to_center_dist < min_dist:
                log.info(
                    f"Rejecting {place_target_name}: too close to pickup object "
                    f"(center distance={center_to_center_dist:.3f}m < min={min_dist:.3f}m)"
                )
                continue

            self.place_receptacle_name = place_target_name
            place_target_found = True
            break

        return place_target_found

    def _filter_place_target(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        place_target_name: str,
    ) -> bool:
        """No size constraint for 'next to' placement."""
        return True

    def _sample_task(self, env: CPUMujocoEnv) -> PickAndPlaceNextToTask:
        """Sample task and return PickAndPlaceNextToTask."""
        # Reuse all parent logic, just return different task type
        super()._sample_task(env)
        return PickAndPlaceNextToTask(env, self.config)
