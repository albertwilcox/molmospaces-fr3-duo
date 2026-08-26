import logging

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import mujoco

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.planner.astar_planner import AStarPlanner
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.util_samplers.navgoal_sampler import NavGoalSampler
from molmo_spaces.utils.linalg_utils import normalize_ang_error
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)


"""
This planner policy relies on the computation of a sparse set of (x,y) waypoints
from the AStarPlanner. It then builds a navigation plan by interleaving rotation
and translation phases with interpolated waypoints (either by slerp for rotation
phases, or by linear interpolation for translation ones).

Note: Even though rotation phases could be ideally skipped for waypoints that are
colinear (i.e., interpolated between two spatially distant planner waypoints), as
the agent orientation is constant, we still allow a correction, since the controller
can deviate from the ideal plan. We introduce corrections by means of two mechanisms:
  1. Enforcing some intermediate waypoints between spatially distant ones, regardless
     of Euclidean distance (via `path_interpolation_density`, which can be kept low,
     e.g. 1)
  2. Enforcing intermediate waypoint to keep consecutive ones under a limit distance
     (via `path_max_inter_waypoint_dist`, which should be kept low enough to prevent
     too much overshooting by the controller)

Note 2: For rotation, we limit the maximal arc length via `path_max_inter_waypoint_angle`,
which we set by default to 10 degrees. This, combined with a fix in the holonomic
base control to express the current yaw according to the intended motion direction to
prevent wrapping errors, leads to smooth rotations.

Note 3: the current replanning heuristic is brittle, so we can just switch it off by
e.g. setting `plan_fail_after_waypoint_steps` to a value larger than the task horizon.

Some TODOs:
 - Use joint_pos_rel to decide upon failure to complete previous action
 - Select the nearest plannable random location near target instead of first one with valid plan
"""


class AStarPlannerPolicy(PlannerPolicy):
    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)

        self._target_pos_quat = None
        self._nav_goal_sampler = None

        self.config.policy_config.planner_config.agent_radius = (
            getattr(self.config.policy_config, "nav_planning_agent_radius", None)
            if getattr(self.config.policy_config, "nav_planning_agent_radius", None)
            is not None
            else self.config.task_sampler_config.robot_safety_radius
        )

        self.nav_planner = AStarPlanner(
            self.config.policy_config.planner_config, self.task.env.current_model_path
        )

        # Fallback planner at a SMALLER agent radius. The primary nav radius
        # (``nav_planning_agent_radius``, e.g. 0.40) is inflated to clear the
        # stowed arm from walls (fixing the mid-nav wedge), but in tight houses
        # that inflation can DISCONNECT the free-space graph so no path exists to
        # a valid goal ("[A* PLAN ATTEMPT FAIL] no valid path found" for every
        # goal from a fixed start). When the primary planner returns no path we
        # retry with this thinner planner (down to ``robot_safety_radius``), which
        # recovers the pre-inflation connectivity. Built lazily on first use.
        self._nav_planner_fallback: AStarPlanner | None = None
        fb_radius = getattr(
            self.config.policy_config, "nav_planning_fallback_agent_radius", None
        )
        if fb_radius is None:
            fb_radius = float(self.config.task_sampler_config.robot_safety_radius)
        primary_radius = float(self.config.policy_config.planner_config.agent_radius)
        self._nav_fallback_radius = (
            float(fb_radius) if float(fb_radius) < primary_radius else None
        )

        self._current_waypoint = 0
        self._reached_waypoints = 0
        self._nav_plan = None
        self._target_pos_quat = None
        self._dists_to_waypoint = []
        self._retries_left = self.config.policy_config.plan_max_retries
        self._target_object = None
        self._candidate_objs = None
        self._skipped_candidates = set()
        self._replan_after = None

        self.robot_view = task.env.current_robot.robot_view

    def planners(self):
        return self.nav_planner

    def _fallback_motion_plan(self, target_pos, robot_view, max_goal_snap_m=None):
        """Plan with a thinner agent radius when the primary (inflated) planner
        finds no path. Returns world waypoints or None. Lazily builds a second
        planner at ``self._nav_fallback_radius`` so the primary planner's cached
        map/graph are untouched."""
        if self._nav_fallback_radius is None:
            return None
        if self._nav_planner_fallback is None:
            import copy as _copy

            fb_cfg = _copy.copy(self.config.policy_config.planner_config)
            fb_cfg.agent_radius = self._nav_fallback_radius
            self._nav_planner_fallback = AStarPlanner(
                fb_cfg, self.task.env.current_model_path
            )
        try:
            return self._nav_planner_fallback.motion_plan(
                target_pos, robot_view, max_goal_snap_m=max_goal_snap_m
            )
        except ValueError:
            return None

    def reset(self):
        self._current_waypoint = 0
        self._reached_waypoints = 0
        self._nav_plan = None
        self._target_pos_quat = None
        self._dists_to_waypoint = []
        self._retries_left = self.config.policy_config.plan_max_retries
        self._target_object = None
        self._candidate_objs = None
        self._skipped_candidates = set()
        self._replan_after = None
        self._nav_goal_override_tried = False
        self._using_override_goal = False
        self.nav_planner.blacklist.clear()

    @property
    def candidate_objs(self) -> list[MlSpacesObject]:
        if self._candidate_objs is None:
            batch_idx = self.task.env.current_batch_index
            self._candidate_objs = self.task.nav_objs[batch_idx]

        return self._candidate_objs

    def skip_candidate(self, obj_name):
        self._skipped_candidates.add(obj_name)

    @property
    def target_object(self) -> MlSpacesObject:
        """Get the nearest navigation target object for the current batch."""
        if (
            self._target_object is None
            or not self.config.policy_config.plan_stick_to_original_target
            or self._target_object.name in self._skipped_candidates
        ):
            if len(self.candidate_objs) == 1:
                self._target_object = self.candidate_objs[0]
            else:
                # If more than one candidate, return the nearest remaining one
                batch_idx = self.task.env.current_batch_index
                priority = self.task.get_nav_object_priority(batch_idx)
                for obj in priority:
                    if obj.name in self._skipped_candidates:
                        continue
                    else:
                        self._target_object = obj
                        break
                else:
                    # Fallback: just pick the nearest
                    self._target_object = priority[0] if priority else None

        return self._target_object

    @property
    def nav_goal_sampler(self) -> NavGoalSampler:
        if self._nav_goal_sampler is None:
            self._nav_goal_sampler = NavGoalSampler(
                self.nav_planner.map,
                check_target_in_view=False,
                camera_name="head_camera",
                distance_threshold=self.config.policy_config.nav_goal_distance_threshold,
            )

        return self._nav_goal_sampler

    @property
    def target_pos_quat(self):
        if self._target_pos_quat is None:
            # Feasibility-verified goal override (Avenue A): if the task provides a
            # base pose the sampler already proved is manip-feasible, drive to it
            # instead of re-sampling a standoff goal at the closest navigable cell.
            # Used at most once per nav phase; if unreachable, ``nav_plan`` clears
            # ``_target_pos_quat`` and this falls through to goal sampling below.
            override = getattr(self.task, "nav_goal_override", None)
            if override is not None and not getattr(self, "_nav_goal_override_tried", False):
                self._nav_goal_override_tried = True
                self._using_override_goal = True
                self._target_pos_quat = (np.asarray(override[0]), np.asarray(override[1]))
                if hasattr(self.task, "set_planned_nav_goal"):
                    self.task.set_planned_nav_goal(override[0], override[1])
                log.info(
                    f"[A* PLAN] Using feasibility-verified nav goal override at "
                    f"({override[0][0]:.2f}, {override[0][1]:.2f})"
                )
                return self._target_pos_quat

            self.nav_goal_sampler.set_target(self.target_object)
            self.nav_goal_sampler.set_robot_view(self.robot_view)
            self._using_override_goal = False
            cfg = self.config.policy_config
            succ_thr = self.config.task_config.succ_pos_threshold
            margin = getattr(cfg, "nav_goal_success_margin", 0.0)
            target_xy = np.asarray(self.target_object.position)[:2]

            # Stage 1: draw several standoff candidates and keep the one closest
            # to the object centre, accepting early once a candidate lands inside
            # the success ring (succ_pos_threshold - margin). This converts the
            # "arrived but parked just outside the success distance" near-misses
            # into successes, and otherwise still picks the best reachable pose.
            best_candidate = None
            best_center_dist = np.inf
            for attempt in range(cfg.nav_goal_max_attempts):
                candidate = self.nav_goal_sampler.sample()
                if candidate is None:
                    continue
                if cfg.nav_check_goal_visibility and not self._goal_is_visible(*candidate):
                    log.info(f"[A* PLAN] Rejected goal on attempt {attempt}: target not in view")
                    continue
                center_dist = float(np.linalg.norm(np.asarray(candidate[0])[:2] - target_xy))
                if center_dist < best_center_dist:
                    best_center_dist = center_dist
                    best_candidate = candidate
                if center_dist <= max(succ_thr - margin, 0.0):
                    break

            if best_candidate is not None:
                self._target_pos_quat = best_candidate
                # Publish the committed pre-grasp goal to the task so the
                # goal-pose-reaching success criterion (succ_use_goal_pose) can
                # judge arrival on this pose rather than object-centre distance.
                if hasattr(self.task, "set_planned_nav_goal"):
                    self.task.set_planned_nav_goal(best_candidate[0], best_candidate[1])
                log.info(
                    f"[A* PLAN] Selected goal {best_center_dist:.2f}m from target centre"
                    f" (success threshold {succ_thr:.2f}m)"
                )

        return self._target_pos_quat

    def _goal_is_visible(self, position: np.ndarray, quaternion: np.ndarray) -> bool:
        """Feasibility check (#1): would the target be visible from a candidate
        goal pose?

        Temporarily moves the base to the candidate pose, refreshes the nav-camera
        frame, and renders a segmentation frame to measure how much of the target
        is in view (reusing the same visibility definition the task uses to judge
        success). Rejects "reached-but-blind" goals. The base pose is always
        restored.
        """
        cfg = self.config.policy_config
        target = self.target_object
        if target is None:
            return True

        env = self.task.env
        robot_view = self.robot_view
        cam_name = cfg.visibility_camera_name

        saved_pose = robot_view.base.pose.copy()
        try:
            robot_view.base.pose = pos_quat_to_pose_mat(position, quaternion)
            mujoco.mj_forward(robot_view.mj_model, robot_view.mj_data)
            env.camera_manager.registry.update_all_cameras(env)

            visibility = env.check_visibility(cam_name, target.name)
            if isinstance(visibility, dict):
                visibility = visibility.get(target.name, 0.0)
            return float(visibility) > cfg.visibility_min_fraction
        except Exception as exc:  # never let the feasibility gate crash planning
            log.warning(f"[A* PLAN] Visibility check errored ({exc}); accepting goal")
            return True
        finally:
            robot_view.base.pose = saved_pose
            mujoco.mj_forward(robot_view.mj_model, robot_view.mj_data)
            try:
                env.camera_manager.registry.update_all_cameras(env)
            except Exception:
                pass

    def _world_clearance(self, xy: np.ndarray) -> float:
        """Clearance (metres beyond the inflated footprint) at a world (x, y)
        point, read from the planner's distance transform."""
        planner = self.nav_planner
        dt = planner.dt
        px = planner.map.pos_m_to_px(np.array([xy[0], xy[1], 0.0]))
        row = int(np.clip(np.floor(px[0] / planner.downscale), 0, dt.shape[0] - 1))
        col = int(np.clip(np.floor(px[1] / planner.downscale), 0, dt.shape[1] - 1))
        return float(dt[row, col] * planner.grid_spacing)

    @staticmethod
    def _nearest_point_on_polyline(point: np.ndarray, polyline: np.ndarray) -> np.ndarray:
        """Nearest point to ``point`` on the polyline defined by ``polyline``."""
        point = np.asarray(point, dtype=float)
        best = np.asarray(polyline[0], dtype=float)
        best_d2 = np.inf
        for i in range(len(polyline) - 1):
            a = np.asarray(polyline[i], dtype=float)
            b = np.asarray(polyline[i + 1], dtype=float)
            ab = b - a
            denom = float(ab @ ab)
            t = 0.0 if denom == 0.0 else float(np.clip((point - a) @ ab / denom, 0.0, 1.0))
            proj = a + t * ab
            d2 = float((point - proj) @ (point - proj))
            if d2 < best_d2:
                best_d2 = d2
                best = proj
        return best

    def _repair_clearance(self, points: np.ndarray, safe_polyline: np.ndarray) -> np.ndarray:
        """Snap any low-clearance smoothed point back onto the clearance-safe A*
        polyline, leaving safe points untouched (clearance-aware smoothing, #2)."""
        min_clear = self.config.policy_config.nav_smooth_min_clearance
        repaired = np.asarray(points, dtype=float).copy()
        for i in range(len(repaired)):
            if self._world_clearance(repaired[i]) <= min_clear:
                repaired[i] = self._nearest_point_on_polyline(repaired[i], safe_polyline)
        return repaired

    def stop_plan(self, waypoints: np.ndarray) -> np.ndarray:
        r = self.config.policy_config.path_min_dist_to_target_center

        if r == 0.0:
            return waypoints

        cc = self.target_object.position[:2]

        # 1. find first waypoint entering circle and not leaving again
        last_out = None
        for i in reversed(range(len(waypoints))):
            if np.linalg.norm(waypoints[i] - cc) > r:
                last_out = i
                break
        else:
            # all waypoints were under r, so use the first two waypoints only
            return waypoints[:2]

        # if all waypoints are further than r, keep them all
        if last_out == len(waypoints) - 1:
            return waypoints

        # If not, intersect circumference around object center and last segment
        # (x-c)^T(x-c) = r^2
        # with x = s + alpha d
        # resulting in alpha^2 * (d^Td) + alpha * [2 d^T(s-c)] +[(s-c)^T(s-c) - r^2] = 0
        # for convenience, we make d a unitary direction

        segment = waypoints[last_out : last_out + 2]
        s = segment[0]

        d = segment[1] - s
        d /= np.linalg.norm(d)  # so a == 1 in the 2nd order equation

        sc = s - cc
        b = 2 * d @ sc
        c = np.linalg.norm(sc) ** 2 - r**2

        # Discriminant should always be positive, as the two points differ
        # in their inclusion in the circle with given radius
        # (we enforce it's at least non-negative)
        disc = max(b**2 - 4 * c, 0)

        # We keep the smallest (entering) solution (minus sign)
        # the relative displacement from the last waypoint outside along the
        # direction to the first one inside needs to be positive
        # (we enforce it's at least non-negative)
        alpha = max((-b - np.sqrt(disc)) / 2, 0)

        intersection = s + alpha * d
        return np.concatenate([waypoints[: last_out + 1], intersection[None, :]])

    def max_dist_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        assert waypoints.shape == (2, 2)

        direction = waypoints[-1] - waypoints[0]

        dist = np.linalg.norm(direction)
        num_points = int(np.ceil(dist / self.config.policy_config.path_max_inter_waypoint_dist))
        if num_points <= 1:
            return waypoints[1:]

        stops = np.linspace(0, 1, num_points + 1)[1:]
        return waypoints[:1] + direction[None, :] * stops[:, None]

    def max_angle_waypoints(self, angles: np.ndarray) -> np.ndarray:
        assert angles.shape == (2, 1)

        angle = float(abs(normalize_ang_error((angles[1] - angles[0]).item())))
        num_points = int(np.ceil(angle / self.config.policy_config.path_max_inter_waypoint_angle))
        if num_points <= 1:
            # Enofrce always at least one orientation correction
            return angles[1:]

        steps = np.linspace(0, 1, num_points + 1)[1:]
        r0 = R.from_euler("z", angles[0], degrees=False)
        r1 = R.from_euler("z", angles[1], degrees=False)
        rots = Slerp([0, 1], R.concatenate([r0, r1]))(steps)
        new_angles = rots.as_euler("xyz", degrees=False)[:, 2:]

        return new_angles

    def interpolate_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Interpolate waypoints between each pair of waypoints.

        Args:
            waypoints: original waypoints array, shape (N, 2)

        Returns:
            interpolated waypoints array
        """
        density = self.config.policy_config.path_interpolation_density
        if density <= 0 or waypoints is None or len(waypoints) <= 1:
            return waypoints

        # Use np.linspace for faster vectorized interpolation
        segments = []
        for i in range(len(waypoints) - 1):
            # Create density+2 points from waypoints[i] to waypoints[i+1], excluding endpoint
            t = np.linspace(0, 1, density + 2, endpoint=False)[1:]  # exclude start point
            segment = waypoints[i] + t[:, np.newaxis] * (waypoints[i + 1] - waypoints[i])
            segments.append(segment)
        segments.append(waypoints[-1:])  # add final waypoint

        return np.vstack([waypoints[0:1]] + segments)

    def _nav_goal_override_active(self) -> bool:
        """True while the current nav phase is driving to a feasibility-verified
        standoff (nav goal override). In that case the object-centre truncation in
        ``stop_plan`` (radius ``path_min_dist_to_target_center``) must be skipped:
        it cuts the path where it first enters the object circle -- often on a
        DIFFERENT bearing than the standoff -- leaving the base parked short of
        the standoff on the wrong side. That divergence makes the PICK/PLACE build
        fail at the navigated pose and the standoff snap cross a wall (0
        candidates). Keeping the full path lets the base reach the verified
        standoff itself."""
        return (
            getattr(self.task, "nav_goal_override", None) is not None
            and getattr(self, "_nav_goal_override_tried", False)
        )

    def build_policy_plan(self, world_waypoints):
        if not self._nav_goal_override_active():
            world_waypoints = self.stop_plan(world_waypoints)

        # the first difference computes theta from first to second waypoint
        pos_deltas = world_waypoints[1:] - world_waypoints[:-1]
        # Here we have the thetas from waypoint i to i+1
        thetas = np.arctan2(pos_deltas[:, 1], pos_deltas[:, 0])[:, None]

        combined_waypoints = []

        # First, we orient toward the 1st waypoint from the 0-th waypoint
        start_theta = self.robot_view.get_noop_ctrl_dict(["base"])["base"][2]
        for theta in self.max_angle_waypoints(np.stack([[start_theta], thetas[0]])):
            combined_waypoints.append(np.concatenate((world_waypoints[0], theta)))

        for i in range(1, len(world_waypoints) - 1):
            # We move towards i-th waypoint from the (i-1)-th waypoint
            for waypoint in self.max_dist_waypoints(world_waypoints[i - 1 : i + 1]):
                combined_waypoints.append(np.concatenate((waypoint, thetas[i - 1])))

            # We orient towards (i+1)-th waypoint from the i-th waypoint
            for theta in self.max_angle_waypoints(thetas[i - 1 : i + 1]):
                combined_waypoints.append(np.concatenate((world_waypoints[i], theta)))

        # First arrive at final position with last movement direction
        for waypoint in self.max_dist_waypoints(world_waypoints[-2:]):
            combined_waypoints.append(np.concatenate((waypoint, thetas[-1])))

        # Then rotate to face the target
        final_pos = world_waypoints[-1]
        target_pos = self.target_object.position[:2]
        final_theta = np.arctan2(target_pos[1] - final_pos[1], target_pos[0] - final_pos[0])
        for theta in self.max_angle_waypoints(np.stack([thetas[-1], [final_theta]])):
            combined_waypoints.append(np.concatenate((final_pos, theta)))

        return np.array(combined_waypoints)

    @property
    def nav_plan(self):
        if self._nav_plan is None:
            total_attempts = 0
            for candidate_attempt in range(
                max(len(self.candidate_objs) - len(self._skipped_candidates), 1)
            ):
                for pose_attempt in range(5):
                    total_attempts += 1
                    if self.target_pos_quat is None:
                        log.info(
                            "[A* PLAN ATTEMPT FAIL] target_pos_quat is None - NavGoalSampler failed to find valid goal position"
                        )
                        break
                    else:
                        # For the feasibility-verified override goal, guard
                        # against the planner silently snapping the goal into a
                        # nearby wall (the goal sits inside the inflation band).
                        # When it would snap too far, the primary returns None and
                        # we recover the true standoff via the thinner fallback,
                        # whose graph typically contains it. Sampled goals snap to
                        # the object ring by design, so they are left unguarded.
                        snap_guard = (
                            getattr(
                                self.config.policy_config,
                                "nav_goal_override_max_snap_m",
                                None,
                            )
                            if getattr(self, "_using_override_goal", False)
                            else None
                        )
                        world_waypoints = None
                        try:
                            world_waypoints = self.nav_planner.motion_plan(
                                self.target_pos_quat[0],
                                self.robot_view,
                                max_goal_snap_m=snap_guard,
                            )
                        except ValueError as e:
                            if "starting position" in str(e):
                                self._nav_plan = None
                                return self._nav_plan

                        if world_waypoints is None:
                            # Primary (inflated-radius) planner found no path (or
                            # refused an over-snapped override goal). Retry with
                            # the thinner fallback planner before giving up: the
                            # inflation may have disconnected the free-space graph
                            # in a tight house, or the standoff sits just inside
                            # the inflation band.
                            world_waypoints = self._fallback_motion_plan(
                                self.target_pos_quat[0],
                                self.robot_view,
                                max_goal_snap_m=snap_guard,
                            )
                            if world_waypoints is not None:
                                log.info(
                                    "[A* PLAN] primary radius found no path; "
                                    f"recovered via fallback radius "
                                    f"{self._nav_fallback_radius:.2f}m."
                                )

                        if world_waypoints is None:
                            robot_pos = self.robot_view.base.pose[:3, 3]
                            target_pos = self.target_pos_quat[0]
                            log.info(
                                f"[A* PLAN ATTEMPT FAIL] A* pathfinding failed - no valid path found. "
                                f"Robot pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                                f"Target pos: ({target_pos[0]:.2f}, {target_pos[1]:.2f})"
                            )
                            self._target_pos_quat = None
                            continue
                        else:
                            if self.config.policy_config.path_interpolation_density > 0:
                                world_waypoints = self.interpolate_waypoints(world_waypoints)

                            world_waypoints = self.build_policy_plan(world_waypoints)
                            self._nav_plan = world_waypoints
                            self._current_waypoint = 0
                            self._dists_to_waypoint = []

                            log.info(
                                f"[A* PLAN OK] Path planned successfully with {len(world_waypoints)} waypoints"
                                f" after {pose_attempt + 1} pose samples"
                                f" in the {candidate_attempt + 1}-th highest priority candidate"
                                f" (total {total_attempts} plan attempts)"
                            )
                            break

                if self._nav_plan is None:
                    # Ignore current candidate for future plan attempts
                    self.skip_candidate(self.target_object.name)
                else:
                    break
            else:
                log.warning("[A* PLAN FAIL] no valid trajectory found")

        return self._nav_plan

    def current_waypoint(self):
        if self._current_waypoint < len(self.nav_plan):
            cur_distance = self.robot_view.distance_to(
                ["base"], self.nav_plan[self._current_waypoint]
            )
            pose = self.robot_view.base.pose
            angle = R.from_matrix(pose[:3, :3]).as_euler("xyz")[2:3]
            delta = pose[:2, 3]
            pose = np.concatenate((delta, angle))

            log.debug(
                f"Steps {self.task.num_steps_taken()}"
                f" Retries left {self._retries_left}"
                f" Waypoint {self._reached_waypoints}/{len(self.nav_plan) + self._reached_waypoints - self._current_waypoint}"
                f" {np.round(self.nav_plan[self._current_waypoint], 3)}"
                f" Pose {np.round(pose, 3)}"
                f" Dist {cur_distance:3f}"
            )

            if self.robot_view.is_close_to(["base"], self.nav_plan[self._current_waypoint]):
                self._reached_waypoints += 1
                self._dists_to_waypoint = []

                if self._replan_after is None:
                    self._current_waypoint += 1
                else:
                    if self._replan_after <= 1:
                        self._replan_after = None
                        self._nav_plan = None
                        self._target_pos_quat = None
                        if self.nav_plan is None:
                            log.warning("Terminating due to failure to replan.")
                            return None
                    else:
                        self._replan_after -= 1
                        self._current_waypoint -= 1

            elif (
                len(self._dists_to_waypoint)
                > self.config.policy_config.plan_fail_after_waypoint_steps
                and min(self._dists_to_waypoint) - cur_distance
                <= self.config.policy_config.plan_fail_max_dist_delta
            ):
                self.nav_planner.blacklist.append(self.robot_view.base.pose[:3, 3].copy())
                self.nav_planner.apply_black_list()
                if self._retries_left > 0:
                    if self._replan_after is None:

                        def waypoints_until_different_location():
                            back = 0
                            while (
                                np.linalg.norm(
                                    self.nav_plan[self._current_waypoint][:2]
                                    - self.nav_plan[self._current_waypoint - back][:2]
                                )
                                < 0.25  # TODO make this a config param
                            ):
                                if self._current_waypoint - back > 0:
                                    back += 1
                                else:
                                    break

                            return back

                        self._retries_left -= 1
                        self._replan_after = waypoints_until_different_location()
                        self._dists_to_waypoint = []
                        self._current_waypoint = max(self._current_waypoint - 1, 0)
                        log.warning(
                            f"Replanning requested after returning to the previous {self._replan_after} waypoints"
                            f" due to failure to progress with distance {cur_distance:.3f}"
                            f" to waypoint with {self._retries_left} retries left."
                        )
                    else:
                        log.warning(
                            f"Terminating due to failure to return to previous waypoint"
                            f" with {self._replan_after} missing return waypoints"
                        )
                        return None
                else:
                    log.warning(
                        f"Terminating due to failure to progress with distance {cur_distance:.3f} to waypoint"
                        f" and no plan retries left."
                    )
                    return None

            else:
                self._dists_to_waypoint.append(cur_distance)

        if self._current_waypoint < len(self.nav_plan):
            return self.nav_plan[self._current_waypoint]

        return None

    def get_action(self, observation):
        if self.nav_plan is None:
            # No plan possible, finish task immediately
            log.warning(
                f"[A* DONE] Planning failed - terminating episode at step {self.task.num_steps_taken()}"
                f" with {self._reached_waypoints} reached waypoints."
                f" Reason: A* could not find a valid path (see earlier PLAN FAIL logs for details)"
            )
            return self._build_done_action()

        # get next waypoint in the planned trajectory
        waypoint = self.current_waypoint()

        if waypoint is None:
            # All waypoints reached - navigation complete
            log.info(
                f"[A* DONE] Navigation complete - reached {self._reached_waypoints} waypoints"
                f" in {self.task.num_steps_taken()} steps."
            )
            return self._build_done_action()

        # Still navigating - return action to reach next waypoint
        return self._build_navigation_action(waypoint)

    def _build_done_action(self):
        """Build action to signal episode completion."""
        return {**self.robot_view.get_noop_ctrl_dict(["base"]), "done": True}

    def _build_navigation_action(self, waypoint):
        """Build action to navigate toward the given waypoint."""
        return {"done": False, "base": waypoint}


class AStarSmoothPlannerPolicy(AStarPlannerPolicy):
    def build_policy_plan(self, world_waypoints):
        if not self._nav_goal_override_active():
            world_waypoints = self.stop_plan(world_waypoints)

        plan_length = sum(
            np.linalg.norm(world_waypoints[it] - world_waypoints[it - 1])
            for it in range(1, len(world_waypoints))
        )
        num_points = 2 * int(
            np.ceil(plan_length / self.config.policy_config.path_max_inter_waypoint_dist)
        )

        tck, u = splprep(world_waypoints.transpose(), s=1e-5)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)

        # First derivatives
        dx_du, dy_du = splev(u_new, tck, der=1)
        # Tangent angle (radians)
        thetas = np.arctan2(dy_du, dx_du)
        # TODO handle large theta deltas

        if self.config.policy_config.nav_smooth_clearance_repair:
            # Clearance-aware smoothing (#2): pull any smoothed point that the
            # spline pushed too close to (or into) an obstacle back onto the
            # clearance-safe A* polyline, then recompute tangents from the
            # repaired points so orientations stay consistent.
            smoothed = np.stack([x_new, y_new], axis=1)
            smoothed = self._repair_clearance(smoothed, world_waypoints)
            x_new, y_new = smoothed[:, 0], smoothed[:, 1]
            tangents = np.gradient(smoothed, axis=0)
            thetas = np.arctan2(tangents[:, 1], tangents[:, 0])

        combined_waypoints = []

        # First, we orient toward the 1st waypoint from the 0-th waypoint
        start_theta = self.robot_view.get_noop_ctrl_dict(["base"])["base"][2]
        for theta in self.max_angle_waypoints(np.stack([start_theta, thetas[0]])[:, None]):
            combined_waypoints.append(np.concatenate((world_waypoints[0], theta)))

        for cur_x, cur_y, cur_theta in zip(x_new, y_new, thetas):
            combined_waypoints.append(np.stack((cur_x, cur_y, cur_theta)))

        # Then rotate to face the target
        final_pos = world_waypoints[-1]
        target_pos = self.target_object.position[:2]
        final_theta = np.arctan2(target_pos[1] - final_pos[1], target_pos[0] - final_pos[0])
        for theta in self.max_angle_waypoints(np.stack([thetas[-1], final_theta])[:, None]):
            combined_waypoints.append(np.concatenate((final_pos, theta)))

        return np.array(combined_waypoints)


class PurePursuitNavToObjPolicy(AStarSmoothPlannerPolicy):
    """Closed-loop pure-pursuit follower over the A*/smoothed reference path.

    Reuses the parent's global planning, success-threshold-aware goal selection
    and B-spline smoothing, but replaces the open-loop, advance-on-proximity
    waypoint consumption (``current_waypoint``) with a look-ahead "carrot"
    computed from the *current* base pose every step:

      1. Project the current base (x, y) onto the smoothed reference polyline and
         measure the arc-length travelled.
      2. Place a carrot ``pursuit_lookahead_m`` further along the path and command
         the holonomic base toward it (heading = bearing to the carrot).
      3. Near the end, hold the final point and rotate to face the target, then
         finish; abort if arc-length stops progressing for ``pursuit_max_stall_steps``.

    Because progress is measured by arc-length projection rather than by reaching
    discrete setpoints, the base never wedges on a waypoint the position servo
    cannot hit exactly -- the dominant open-loop stall mode.
    """

    def reset(self):
        super().reset()
        self._ref_xy = None
        self._final_face_theta = None
        self._cum = None
        self._max_s_reached = 0.0
        self._stall_steps = 0
        self._terminal_steps = 0
        self._align_best_ang_err = np.inf
        self._align_no_improve = 0
        self._cmd_yaw = None  # continuous slew-limited commanded yaw (see _slew_yaw)
        self._cmd_yaw_rate = 0.0  # commanded yaw velocity (rad/s), for accel limiting
        self._cmd_xy = None  # continuous speed-limited commanded (x, y) (see _slew_xy)

    def build_policy_plan(self, world_waypoints):
        # Build the parent's (x, y, theta) plan, then derive the spatial reference
        # polyline the carrot follower tracks (dedupe the in-place rotation phases
        # that keep x,y constant) and cache its cumulative arc-length.
        plan = super().build_policy_plan(world_waypoints)
        xy = np.asarray(plan)[:, :2]
        keep = np.concatenate([[True], np.any(np.abs(np.diff(xy, axis=0)) > 1e-6, axis=1)])
        ref_xy = xy[keep]
        if len(ref_xy) < 2:
            ref_xy = xy[:1] if len(xy) else np.zeros((1, 2))
        self._ref_xy = ref_xy
        self._final_face_theta = float(np.asarray(plan)[-1, 2])
        seg = np.linalg.norm(np.diff(self._ref_xy, axis=0), axis=1)
        self._seg = seg
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])
        self._max_s_reached = 0.0
        self._stall_steps = 0
        self._terminal_steps = 0
        self._align_best_ang_err = np.inf
        self._align_no_improve = 0
        self._cmd_yaw = None
        self._cmd_yaw_rate = 0.0
        self._cmd_xy = None
        # Publish the ACTUAL pre-grasp goal pose the follower drives to (plan
        # endpoint + final facing) so the task's goal-pose-reaching success
        # criterion judges arrival on the pose really tracked -- not the raw goal
        # sampler quaternion, whose facing convention differs from the base frame.
        if hasattr(self.task, "set_planned_nav_goal_pose") and len(self._ref_xy):
            self.task.set_planned_nav_goal_pose(self._ref_xy[-1], self._final_face_theta)
        return plan

    def _current_yaw(self) -> float:
        return float(R.from_matrix(self.robot_view.base.pose[:3, :3]).as_euler("xyz")[2])

    def _slew_yaw(self, desired: float) -> float:
        """Rate- and acceleration-limited commanded base yaw toward ``desired``.

        The raw commanded heading can jump discontinuously (the one-step snap to
        the final-facing angle at the terminal phase, or sharp bends in the
        reference path). Because the base is an absolute-position servo, such a
        setpoint jump is applied aggressively within one policy step, producing
        very high yaw jerk. We instead drive an internal continuous commanded yaw
        toward ``desired`` with a velocity cap (``pursuit_max_yaw_rate_rad_s``)
        and, on top of it, an acceleration cap (``pursuit_max_yaw_accel_rad_s2``)
        so the yaw velocity ramps up/down smoothly and decelerates to rest at the
        target without overshoot -- a trapezoidal profile. The result is a
        C1-smooth yaw command (a much better imitation-learning target) and
        smoother base motion. The ``base.ctrl`` setter re-expresses the wrapped
        setpoint in a continuous frame, so wrapping here is safe.
        """
        cfg = self.config.policy_config
        max_rate = float(getattr(cfg, "pursuit_max_yaw_rate_rad_s", 0.0))
        if max_rate <= 0.0:  # disabled -> pass through
            return float(normalize_ang_error(desired))
        if self._cmd_yaw is None:  # seed from the actual heading on first command
            self._cmd_yaw = self._current_yaw()
            self._cmd_yaw_rate = 0.0
        dt = float(self.config.policy_dt_ms) / 1000.0
        err = float(normalize_ang_error(desired - self._cmd_yaw))
        max_accel = float(getattr(cfg, "pursuit_max_yaw_accel_rad_s2", 0.0))
        if max_accel <= 0.0:
            # Rate cap only: step directly toward the target, bounded by max_rate.
            step = float(np.clip(err, -max_rate * dt, max_rate * dt))
            self._cmd_yaw += step
            self._cmd_yaw_rate = step / dt
            return float(normalize_ang_error(self._cmd_yaw))
        # Trapezoidal profile: target the largest speed we can still decelerate
        # from to arrive at rest on the target (sqrt(2*a*|err|)), capped by
        # max_rate; then bound the per-step change in velocity by max_accel.
        v_desired = np.sign(err) * min(max_rate, float(np.sqrt(2.0 * max_accel * abs(err))))
        dv = float(np.clip(v_desired - self._cmd_yaw_rate, -max_accel * dt, max_accel * dt))
        self._cmd_yaw_rate += dv
        # Don't step past the target within this tick.
        step = float(np.clip(self._cmd_yaw_rate * dt, -abs(err), abs(err)))
        self._cmd_yaw += step
        # If the overshoot guard shortened the step, sync the stored velocity to
        # what actually moved so a stale high rate doesn't cause an acceleration
        # spike on the next tick (keeps the arrival deceleration smooth).
        if abs(step) < abs(self._cmd_yaw_rate * dt):
            self._cmd_yaw_rate = step / dt
        return float(normalize_ang_error(self._cmd_yaw))

    def _slew_xy(self, desired_xy: np.ndarray) -> np.ndarray:
        """Speed-limited commanded base (x, y) toward ``desired_xy``.

        Planar analogue of :meth:`_slew_yaw`. The holonomic base is an absolute-
        POSITION servo whose per-step motion saturates at the actuator velocity
        limit (~2.1 m/s), so commanding the far look-ahead carrot directly drives
        the base far faster than RoboCasa's real nav cruise. We instead advance an
        internal commanded position toward the carrot by at most
        ``pursuit_max_speed_m_s * dt`` per step (this advance rate is what caps the
        recorded per-step displacement, and hence ``action.base_velocity``, at the
        target speed). Separately, the internal setpoint is allowed to LEAD the true
        base pose by up to ``pursuit_max_lead_m`` (>> one step): this preserves the
        servo push-through force ``kp * lead`` so the base can reach cruise speed and
        drive through friction/minor obstacles instead of stalling, while staying
        small enough that unwedge catch-up stays below the teleport detector's cap.
        The cap naturally yields a cruise-then-decelerate profile: far from the goal
        the carrot is always > one step away so the base cruises at the cap, and
        near the goal the shrinking carrot distance lets it brake below the cap.
        Set ``pursuit_max_speed_m_s <= 0`` to pass the carrot through unmodified.
        """
        cfg = self.config.policy_config
        desired_xy = np.asarray(desired_xy, dtype=np.float64)
        max_speed = float(getattr(cfg, "pursuit_max_speed_m_s", 0.0) or 0.0)
        if max_speed <= 0.0:  # disabled -> pass through
            return desired_xy
        dt = float(self.config.policy_dt_ms) / 1000.0
        max_step = max_speed * dt
        # Anti-windup lead cap: allow the setpoint to lead the true pose by more
        # than one step so the servo keeps push-through force AND can reach the
        # cruise speed, but bound it so unwedge catch-up stays sub-teleport.
        max_lead = max(float(getattr(cfg, "pursuit_max_lead_m", 0.0) or 0.0), max_step)
        cur_xy = self.robot_view.base.pose[:2, 3].astype(np.float64)
        if self._cmd_xy is None:  # seed from the actual base position on first command
            self._cmd_xy = cur_xy.copy()
        # Anti-windup: never let the internal setpoint lead the ACTUAL base by more
        # than ``max_lead``. If the base stalls (obstacle) while the carrot keeps
        # advancing, an unclamped setpoint would run arbitrarily far ahead and then
        # command a large jump once the base frees -- exactly the teleport we are
        # removing. Pulling the setpoint back to within ``max_lead`` of the true
        # pose keeps this a bounded velocity command regardless of tracking error.
        lead = self._cmd_xy - cur_xy
        lead_dist = float(np.linalg.norm(lead))
        if lead_dist > max_lead:
            self._cmd_xy = cur_xy + lead * (max_lead / lead_dist)
        delta = desired_xy - self._cmd_xy
        dist = float(np.linalg.norm(delta))
        if dist > max_step:
            delta = delta * (max_step / dist)
        self._cmd_xy = self._cmd_xy + delta
        return self._cmd_xy

    def _project_arclength(self, point: np.ndarray) -> float:
        """Arc-length of the closest point on the reference polyline to ``point``."""
        path = self._ref_xy
        best_s, best_d2 = 0.0, np.inf
        for i in range(len(path) - 1):
            a = path[i]
            ab = path[i + 1] - a
            denom = float(ab @ ab)
            t = 0.0 if denom == 0.0 else float(np.clip((point - a) @ ab / denom, 0.0, 1.0))
            proj = a + t * ab
            d2 = float((point - proj) @ (point - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_s = float(self._cum[i] + t * self._seg[i])
        return best_s

    def _point_at_arclength(self, s: float) -> np.ndarray:
        """Interpolate the reference polyline at arc-length ``s``."""
        cum = self._cum
        s = float(np.clip(s, 0.0, cum[-1]))
        i = int(np.searchsorted(cum, s) - 1)
        i = int(np.clip(i, 0, len(self._seg) - 1))
        seg = self._seg[i]
        t = 0.0 if seg == 0.0 else (s - cum[i]) / seg
        return self._ref_xy[i] + t * (self._ref_xy[i + 1] - self._ref_xy[i])

    def get_action(self, observation):
        # Trigger (lazy) planning + goal selection via the parent machinery.
        if self.nav_plan is None or self._ref_xy is None or len(self._ref_xy) < 2:
            log.warning(
                f"[PurePursuit DONE] No plan/path available at step {self.task.num_steps_taken()}"
            )
            return self._build_done_action()

        cfg = self.config.policy_config
        total = float(self._cum[-1])
        cur_xy = self.robot_view.base.pose[:2, 3]
        s_proj = self._project_arclength(cur_xy)
        remaining = total - s_proj

        # Arc-length stall detection (the base is wedged / not advancing).
        if s_proj > self._max_s_reached + 1e-3:
            self._max_s_reached = s_proj
            self._stall_steps = 0
        else:
            self._stall_steps += 1

        # Terminal phase: at the path end, hold position and align to face target.
        if remaining <= cfg.pursuit_goal_tol_m:
            self._terminal_steps += 1
            final_xy = self._ref_xy[-1]
            ang_err = abs(normalize_ang_error(self._final_face_theta - self._current_yaw()))
            if ang_err <= cfg.pursuit_final_align_tol_rad:
                log.info(
                    f"[PurePursuit DONE] Reached path end in {self.task.num_steps_taken()} steps"
                    f" ({self._reached_waypoints} carrots advanced)."
                )
                return self._build_done_action()
            # Track angular progress: keep turning while the heading error is
            # still shrinking (up to the large ``pursuit_final_align_max_steps``
            # budget), but bail out early once it plateaus for
            # ``pursuit_final_align_stall_steps`` -- the base is wedged and cannot
            # rotate further, so spinning out the full budget only wastes steps.
            if ang_err + 1e-3 < self._align_best_ang_err:
                self._align_best_ang_err = ang_err
                self._align_no_improve = 0
            else:
                self._align_no_improve += 1
            # Bound the final-alignment phase with its OWN budget (not the mid-path
            # stall budget): the absolute-heading position servo slews at a bounded
            # rate, so a large in-place turn needs many steps. Giving up too early
            # parks the base facing away from the target and dooms the grasp/place.
            if (
                self._terminal_steps > cfg.pursuit_final_align_max_steps
                or self._align_no_improve
                > getattr(cfg, "pursuit_final_align_stall_steps", 40)
            ):
                log.warning(
                    f"[PurePursuit DONE] Reached path end but could not align"
                    f" (|ang err|={ang_err:.2f}rad) after {self._terminal_steps} align steps;"
                    f" finishing at step {self.task.num_steps_taken()}."
                )
                return self._build_done_action()
            return {"done": False, "base": np.array([*self._slew_xy(final_xy), self._slew_yaw(self._final_face_theta)])}

        if self._stall_steps > cfg.pursuit_max_stall_steps:
            log.warning(
                f"[PurePursuit DONE] Terminating: no arc-length progress for"
                f" {self._stall_steps} steps at {remaining:.2f}m remaining."
            )
            return self._build_done_action()

        # Look-ahead carrot along the path; command the base POSITION toward it.
        carrot = self._point_at_arclength(s_proj + cfg.pursuit_lookahead_m)
        # Commanded FACING is derived from a separate, longer look-ahead so the
        # holonomic base "looks where it is going" a bit further out instead of
        # chasing the short position carrot's hypersensitive bearing (which caused
        # a large yaw oscillation). Fall back to the position carrot if the longer
        # look-ahead knob is unset.
        head_ahead = float(getattr(cfg, "pursuit_heading_lookahead_m", 0.0) or 0.0)
        head_pt = self._point_at_arclength(s_proj + head_ahead) if head_ahead > cfg.pursuit_lookahead_m else carrot
        head_delta = head_pt - cur_xy
        if np.linalg.norm(head_delta) > 1e-6:
            travel_heading = float(np.arctan2(head_delta[1], head_delta[0]))
        else:
            travel_heading = self._final_face_theta
        heading = self._strafe_heading(travel_heading, remaining)
        self._reached_waypoints += 1
        return {"done": False, "base": np.array([*self._slew_xy(carrot), self._slew_yaw(heading)])}

    def _strafe_heading(self, travel_heading: float, remaining: float) -> float:
        """Commanded base yaw for a carrot-following step.

        Default (strafe disabled): return ``travel_heading`` so the holonomic base
        yaws to face its direction of travel and drives forward, as before.

        Strafe enabled: within ``strafe_face_target_within_m`` arc-length of the
        goal, progressively blend the commanded heading from the travel bearing
        toward the final target-facing heading (``_final_face_theta``), so the base
        sidles/strafes into the standoff while already facing the object instead of
        driving forward and then spinning in place. The blend is gated by a
        feasibility cone (``strafe_max_lateral_angle_rad``): if facing the target
        would require the base to travel more than that angle off its heading
        (target roughly behind the motion), keep facing travel so the base never
        drives blindly backward. ``_slew_yaw`` still rate/accel-limits the result.
        """
        cfg = self.config.policy_config
        if not getattr(cfg, "nav_enable_strafe", False):
            return travel_heading
        activate = float(getattr(cfg, "strafe_face_target_within_m", 0.0))
        if activate <= 0.0 or remaining > activate:
            return travel_heading
        face = self._final_face_theta
        lateral = abs(normalize_ang_error(face - travel_heading))
        if lateral > float(getattr(cfg, "strafe_max_lateral_angle_rad", np.pi)):
            return travel_heading
        alpha = float(np.clip((activate - remaining) / activate, 0.0, 1.0))
        return float(travel_heading + alpha * normalize_ang_error(face - travel_heading))
