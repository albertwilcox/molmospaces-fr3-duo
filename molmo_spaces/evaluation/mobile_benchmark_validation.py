"""Navigation feasibility for frozen mobile pick-and-place episodes."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.planner.astar_planner import AStarPlanner, AStarPlannerConfig
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

if TYPE_CHECKING:
    from molmo_spaces.env.env import CPUMujocoEnv

logger = logging.getLogger(__name__)


def measure_manipulation_visibility(
    episode: EpisodeSpec, env: "CPUMujocoEnv", camera_names: list[str]
) -> dict[str, dict[str, float]]:
    """Measure both manipulation targets without changing the recorded start."""
    if not camera_names:
        raise ValueError("At least one manipulation visibility camera is required")
    base = env.robots[0].robot_view.base
    original_pose = base.pose.copy()
    result: dict[str, dict[str, float]] = {}
    try:
        for stage, pose_key, object_key in (
            ("pickup", "robot_base_pose", "pickup_obj_name"),
            ("placement", "place_robot_base_pose", "place_receptacle_name"),
        ):
            raw_pose = episode.task.get(pose_key)
            if raw_pose is None or len(raw_pose) != 7 or not np.isfinite(raw_pose).all():
                raise ValueError(f"Missing or invalid manipulation pose: {pose_key}")
            pose = np.asarray(raw_pose)
            base.pose = pos_quat_to_pose_mat(pose[:3], pose[3:])
            env.sync_derived_state()
            values = {
                camera: float(env.check_visibility(camera, episode.task[object_key]))
                for camera in camera_names
            }
            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
                raise ValueError(f"Invalid {stage} visibility measurements: {values}")
            result[stage] = values
    finally:
        base.pose = original_pose
        env.sync_derived_state()
    return result


def validate_navigation(
    episode: EpisodeSpec,
    model_path: str | Path,
    agent_radius: float,
    fallback_agent_radius: float | None = None,
) -> float:
    """Return the connected primary or explicitly configured native fallback radius."""
    try:
        _validate_navigation_at_radius(episode, model_path, agent_radius)
    except ValueError as error:
        if fallback_agent_radius is None or fallback_agent_radius >= agent_radius:
            raise
        logger.warning(
            "Primary navigation radius %.3f rejected: %s; checking native fallback %.3f",
            agent_radius, error, fallback_agent_radius,
        )
        _validate_navigation_at_radius(episode, model_path, fallback_agent_radius)
        return fallback_agent_radius
    return agent_radius


def _validate_navigation_at_radius(episode: EpisodeSpec, model_path: str | Path, agent_radius: float) -> None:
    planner = AStarPlanner(AStarPlannerConfig(agent_radius=agent_radius), str(model_path))
    nodes: list[tuple[int, int]] = []
    for key in ("robot_start_pose", "robot_base_pose", "place_robot_base_pose"):
        raw_pose = episode.task.get(key)
        if raw_pose is None:
            raise ValueError(f"Mobile episode is missing {key}")
        pose = np.asarray(raw_pose[:3])
        node = planner.get_discrete_location(pose)
        if node is None:
            raise ValueError(f"{key} is outside navigable free space")
        projected = planner.map.pos_px_to_m(np.asarray([node]) * planner.downscale)[0]
        snap = float(np.linalg.norm(projected[:2] - pose[:2]))
        if snap > 0.15:
            raise ValueError(f"{key} requires {snap:.3f} m navigation-map snapping")
        nodes.append(tuple(node))
    for start, goal in zip(nodes[:-1], nodes[1:], strict=True):
        if not nx.has_path(planner.graph, start, goal):
            raise ValueError("Navigation start, pickup, and placement standoffs are disconnected")
