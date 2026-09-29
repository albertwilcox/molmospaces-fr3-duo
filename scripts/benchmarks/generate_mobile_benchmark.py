"""Sample a small mobile-PnP benchmark without running an expert policy."""

import argparse
import base64
import json
import logging
import pickle
import random
import signal
from collections.abc import Callable
from pathlib import Path

import numpy as np
from create_json_benchmark import frozen_config_to_episode_spec
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from molmo_spaces.configs.camera_configs import CameraSystemConfig, RobotMountedCameraConfig
from molmo_spaces.configs.mobile_pick_and_place_config import MobileFrankaPickAndPlaceConfig
from molmo_spaces.evaluation.benchmark_schema import (
    CameraSpec,
    EpisodeSpec,
    MobilePickAndPlaceTaskSpec,
)
from molmo_spaces.evaluation.mobile_benchmark_validation import (
    measure_manipulation_visibility,
    validate_navigation,
)
from molmo_spaces.tasks.json_eval_task_sampler import camera_spec_to_config
from molmo_spaces.tasks.task import BaseMujocoTask

logger = logging.getLogger(__name__)


class GenerationConfig(BaseModel):
    output_dir: Path
    houses: list[int] = Field(default_factory=lambda: [0, 1, 2])
    seeds: list[int] = Field(default_factory=lambda: [0, 1])
    data_split: str = "val"
    min_pick_navigation_m: float = 1.0
    min_transport_m: float = 0.8
    sample_timeout_s: int = 300
    pickup_types: list[str] = Field(default_factory=list)
    nav_start_radius_m: tuple[float, float] | None = None
    manip_standoff_radius_m: tuple[float, float] | None = None
    mobile_mount_height_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    initial_arm_qpos: list[float] | None = Field(default=None, min_length=7, max_length=7)
    face_pickup_at_start: bool = False
    camera_specs_json: Path | None = None
    nav_visibility_camera: str | None = None
    min_start_visibility: float = Field(default=0.001, gt=0, le=1)
    exclude_enclosed_placement: bool = False
    place_height_range_m: tuple[float, float] | None = None
    manipulation_visibility_cameras: list[str] | None = Field(default=None, min_length=1)
    min_manipulation_visibility: float = Field(default=0.0002, gt=0, le=1)

    @field_validator("place_height_range_m")
    @classmethod
    def validate_place_heights(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if value is not None and (not np.isfinite(value).all() or not 0 <= value[0] < value[1]):
            raise ValueError("Place heights must be finite and satisfy 0 <= minimum < maximum")
        return value

    @field_validator("nav_start_radius_m", "manip_standoff_radius_m")
    @classmethod
    def validate_nav_radius(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and not 0 < value[0] <= value[1]:
            raise ValueError("Placement radii must satisfy 0 < minimum <= maximum")
        return value

    @field_validator("initial_arm_qpos")
    @classmethod
    def validate_arm_qpos(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and not np.isfinite(value).all():
            raise ValueError("Initial arm joints must be finite")
        return value


class SamplingTimeout(BaseException):
    """Escape nested sampler retry loops when the per-candidate budget expires."""


def _timeout(signum: int, frame: object) -> None:
    raise SamplingTimeout("Mobile task sampling exceeded its time budget")


def generate(
    config: GenerationConfig,
    *,
    experiment_config_factory: Callable[
        ..., MobileFrankaPickAndPlaceConfig
    ] = MobileFrankaPickAndPlaceConfig,
    validate_episode: Callable[[EpisodeSpec, BaseMujocoTask], EpisodeSpec] | None = None,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=False)
    (config.output_dir / "generation_config.json").write_text(config.model_dump_json(indent=2))
    episodes: list[EpisodeSpec] = []
    with (config.output_dir / "sampling.jsonl").open("x") as events:
        for house in config.houses:
            for seed in config.seeds:
                random.seed(seed)
                np.random.seed(seed)
                exp = experiment_config_factory(
                    data_split=config.data_split,
                    seed=seed,
                    profile=False,
                    output_dir=str(config.output_dir),
                )
                exp.task_sampler_config.house_inds = [house]
                if config.mobile_mount_height_m is not None:
                    exp.robot_config.base_size[2] = config.mobile_mount_height_m
                if config.initial_arm_qpos is not None:
                    exp.robot_config.init_qpos["arm"] = config.initial_arm_qpos.copy()
                exp.task_sampler_config.samples_per_house = 1
                exp.task_sampler_config.pickup_types = config.pickup_types
                exp.task_sampler_config.max_place_receptacle_sampling_attempts = 8
                exp.task_sampler_config.place_receptacle_resample_tries = 1
                exp.task_sampler_config.place_exclude_enclosed_container_surfaces = (
                    config.exclude_enclosed_placement
                )
                if config.place_height_range_m is not None:
                    (
                        exp.task_sampler_config.elevated_min_height_m,
                        exp.task_sampler_config.elevated_max_height_m,
                    ) = config.place_height_range_m
                if config.nav_start_radius_m is not None:
                    exp.task_sampler_config.nav_start_radius_range = config.nav_start_radius_m
                if config.manip_standoff_radius_m is not None:
                    exp.task_sampler_config.manip_standoff_radius_range = (
                        config.manip_standoff_radius_m
                    )
                exp.task_sampler_config.nav_start_face_target = config.face_pickup_at_start
                if config.camera_specs_json is not None:
                    cameras = TypeAdapter(list[CameraSpec]).validate_json(
                        config.camera_specs_json.read_text()
                    )
                    exp.camera_config = CameraSystemConfig(
                        img_resolution=(256, 256),
                        cameras=[camera_spec_to_config(camera) for camera in cameras],
                    )
                if config.nav_visibility_camera is not None:
                    camera = exp.camera_config.get_camera_by_name(config.nav_visibility_camera)
                    if not isinstance(camera, RobotMountedCameraConfig):
                        raise ValueError(
                            "Navigation visibility requires a configured robot-mounted camera"
                        )
                    camera.visibility_constraints = {
                        "__pickup_object__": config.min_start_visibility
                    }
                    exp.task_sampler_config.nav_start_check_camera_visibility = True
                for camera_name in config.manipulation_visibility_cameras or []:
                    if not isinstance(
                        exp.camera_config.get_camera_by_name(camera_name), RobotMountedCameraConfig
                    ):
                        raise ValueError(
                            f"Manipulation visibility requires a robot-mounted camera: {camera_name}"
                        )
                sampler = exp.task_sampler_config.task_sampler_class(exp)
                task = None
                episode: EpisodeSpec | None = None
                record: dict = {"house": house, "seed": seed}
                previous_handler = signal.signal(signal.SIGALRM, _timeout)
                signal.alarm(config.sample_timeout_s)
                try:
                    task = sampler.sample_task(house_index=house)
                    if task is None:
                        raise RuntimeError("Mobile task sampler returned no task")
                    # Frozen poses include staged, inactive receptacles too.
                    # Preserve their asset declarations so JSON replay is complete.
                    task.config.task_config.added_objects.update(sampler.added_objects)
                    task.reset()
                    if config.nav_visibility_camera is not None:
                        visibility = task.env.check_visibility(
                            config.nav_visibility_camera, task.config.task_config.pickup_obj_name
                        )
                        record["initial_pickup_visibility"] = visibility
                        if visibility < config.min_start_visibility:
                            raise ValueError(
                                f"Initial pickup visibility {visibility} is below the requested minimum"
                            )
                    # Only deserialize the frozen config just produced in this process.
                    frozen = pickle.loads(base64.b64decode(task.frozen_config))
                    episode = frozen_config_to_episode_spec(
                        frozen_config=frozen,
                        obs_scene=task.get_obs_scene(),
                        house_id=house,
                        scene_dataset=exp.scene_dataset,
                        data_split=exp.data_split,
                        source_h5_file="",
                        source_traj_key=f"sample_house_{house}_seed_{seed}",
                        source_episode_length=0,
                        img_resolution=tuple(exp.camera_config.img_resolution),
                        task_horizon_sec=180,
                    )
                    episode.source = None
                    episode.seed = seed
                    MobilePickAndPlaceTaskSpec.model_validate(episode.task)
                    start = np.asarray(episode.task["robot_start_pose"][:2])
                    pick = np.asarray(episode.task["robot_base_pose"][:2])
                    place_pose = episode.task.get("place_robot_base_pose")
                    if place_pose is None:
                        raise ValueError("No feasible placement standoff was sampled")
                    if config.manipulation_visibility_cameras is not None:
                        measured = measure_manipulation_visibility(
                            episode, task.env, config.manipulation_visibility_cameras
                        )
                        record["manipulation_visibility"] = measured
                        for stage, values in measured.items():
                            if max(values.values()) < config.min_manipulation_visibility:
                                raise ValueError(
                                    f"{stage} target is not visible at its standoff: {values}"
                                )
                    radius = sampler._nav_effective_agent_radius()
                    fallback_radius = (
                        exp.policy_config.nav_policy_config.nav_planning_fallback_agent_radius
                    )
                    if fallback_radius is None:
                        fallback_radius = exp.task_sampler_config.robot_safety_radius
                    checked_radius = validate_navigation(
                        episode, task.env.current_model_path, radius, fallback_radius
                    )
                    record.update(
                        navigation_connected=True,
                        navigation_primary_agent_radius_m=radius,
                        navigation_agent_radius_m=checked_radius,
                        navigation_used_fallback=checked_radius < radius,
                    )
                    place = np.asarray(place_pose[:2])
                    nav_m = float(np.linalg.norm(pick - start))
                    carry_m = float(np.linalg.norm(place - pick))
                    record.update(pick_navigation_m=nav_m, transport_m=carry_m)
                    if nav_m < config.min_pick_navigation_m or carry_m < config.min_transport_m:
                        raise ValueError(
                            f"Task does not require both navigation segments: {record}"
                        )
                    if validate_episode is not None:
                        episode = validate_episode(episode, task)
                    episodes.append(episode)
                    record["status"] = "accepted"
                    (config.output_dir / "benchmark.json").write_text(
                        json.dumps([ep.model_dump(mode="json") for ep in episodes], indent=2)
                    )
                    logger.info(
                        "Accepted house=%s seed=%s nav=%.2fm transport=%.2fm",
                        house,
                        seed,
                        nav_m,
                        carry_m,
                    )
                except (Exception, SamplingTimeout) as exc:
                    logger.exception("Sampling failed: house=%s seed=%s", house, seed)
                    record.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    if episode is not None:
                        with (config.output_dir / "rejected_candidates.jsonl").open(
                            "a"
                        ) as rejected:
                            rejected.write(
                                json.dumps({**record, "episode": episode.model_dump(mode="json")})
                                + "\n"
                            )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous_handler)
                    if sampler.env is not None:
                        sampler.env.close()
                    events.write(json.dumps(record) + "\n")
                    events.flush()
    if not episodes:
        raise RuntimeError(f"No valid mobile episodes; see {config.output_dir / 'sampling.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--houses", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--data-split", default="val")
    parser.add_argument("--sample-timeout-s", type=int, default=300)
    parser.add_argument("--pickup-types", nargs="*", default=[])
    parser.add_argument("--nav-start-radius-m", nargs=2, type=float)
    parser.add_argument("--manip-standoff-radius-m", nargs=2, type=float)
    parser.add_argument("--mobile-mount-height-m", type=float)
    parser.add_argument("--initial-arm-qpos", nargs=7, type=float)
    parser.add_argument("--face-pickup-at-start", action="store_true")
    parser.add_argument("--camera-specs-json", type=Path)
    parser.add_argument("--nav-visibility-camera")
    parser.add_argument("--min-start-visibility", type=float, default=0.001)
    parser.add_argument("--exclude-enclosed-placement", action="store_true")
    parser.add_argument("--place-height-range-m", nargs=2, type=float)
    parser.add_argument("--manipulation-visibility-cameras", nargs="+")
    parser.add_argument("--min-manipulation-visibility", type=float, default=0.0002)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    generate(GenerationConfig(**vars(args)))


if __name__ == "__main__":
    main()
