import copy
import logging
import random
import warnings
from collections import defaultdict

import cv2
import numpy as np
import open3d
import torch

log = logging.getLogger(__name__)

from molmo_spaces.utils.mlspaces_types import ExtraDict, PointsDict


def display_colored_pointclouds_with_opencv(
    render_outputs: list[PointsDict | ExtraDict | tuple[PointsDict, ExtraDict]],
    o3d_vis: open3d.visualization.Visualizer | None = None,
    cv2_window_name: str = "Point clouds",
) -> open3d.visualization.Visualizer | None:
    """static visualizations of obs from all threads"""
    if torch.cuda.is_available():
        log.warning("display() called with CUDA available. Ignoring call.")
        return None

    render_outputs = copy.deepcopy(render_outputs)

    nenvs = len(render_outputs)
    max_columns = 4
    nrow = nenvs // max_columns + (1 if nenvs % max_columns != 0 else 0)
    ncol = nenvs // nrow + (1 if nenvs % nrow != 0 else 0)
    assert nrow * ncol >= nenvs

    if o3d_vis is None:
        o3d_vis = open3d.visualization.Visualizer()
        o3d_vis.create_window(width=512, height=512, visible=False)

    frames = []
    for render_out in render_outputs:
        if isinstance(render_out, tuple):
            points_dict = {**render_out[0], **render_out[1]}
        elif isinstance(render_out, dict):
            points_dict = render_out
        else:
            raise NotImplementedError

        o3d_vis.clear_geometries()
        o3d_vis.reset_view_point(True)

        all_points = points_dict.get("all_coord", None)  # assume debug mode
        if all_points is not None:
            full_pcd = open3d.geometry.PointCloud()
            full_pcd.points = open3d.utility.Vector3dVector(all_points)
            all_colors: np.ndarray = points_dict.get("all_color", None)
            if all_colors is not None:
                alpha = 0.25
                full_pcd.colors = open3d.utility.Vector3dVector(
                    alpha * all_colors + (1 - alpha) * np.array([1, 0.7, 0.7])
                )
            else:
                full_pcd.paint_uniform_color([1, 0.7, 0.7])

            o3d_vis.add_geometry(full_pcd)

        points = points_dict["coord"]
        colors = points_dict.get("color", None)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = open3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.7, 1, 0.7])

        o3d_vis.add_geometry(pcd)

        ctr = o3d_vis.get_view_control()
        ctr.set_front([-0.6, 0.0, -0.8])
        ctr.set_up([-0.8, 0.0, 0.6])
        frames.append(np.asarray(o3d_vis.capture_screen_float_buffer(True)))

    task_size = frames[0].shape
    pad = 255 * np.ones(task_size, dtype=np.uint8)
    for _ in range(len(frames), nrow * ncol):
        frames.append(pad)

    img = np.vstack([np.hstack(row) for row in np.array(frames).reshape(nrow, ncol, *task_size)])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow(cv2_window_name, img)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
        pass

    return o3d_vis


def display_semantic_pointcloud_open3d(
    render_outputs: list[tuple[PointsDict, ExtraDict]],
    static: bool = False,
    o3d_vis: open3d.visualization.Visualizer | None = None,
) -> open3d.visualization.Visualizer | None:
    """
    Display point cloud visualizations for one or multiple environments using Open3D.

    Args:
        render_outputs (List[tuple[PointsDict, ExtraDict]]): A list of tuples containing
            point cloud data and extra information for each environment.
        static (bool, optional): Determines the visualization mode.
            If True, displays static visualizations using Open3D's draw_geometries.
            If False, updates an existing visualization window dynamically.
            Defaults to False.
        o3d_vis (Optional[open3d.visualization.Visualizer], optional): An existing
            Open3D visualizer object. If provided, it will be used for dynamic
            visualization. Defaults to None.

    Returns:
        Optional[open3d.visualization.Visualizer]: The Open3D visualizer object if
        dynamic visualization is used, otherwise None.

    Note:
        - Static mode (static=True) creates a new window for each environment and
          visualizes all environments sequentially. The user must close each window
          to proceed to the next.
        - Dynamic mode (static=False) updates an existing visualization window and
          only visualizes the first environment in the render_outputs list.
        - This function will not execute if CUDA is available to avoid potential
          conflicts with GPU operations.
    """

    if torch.cuda.is_available():
        log.warning("display() called with CUDA available. Ignoring call.")
        return

    render_outputs = copy.deepcopy(render_outputs)

    rnd = random.Random(0)
    colors_map = defaultdict(lambda: [c / 255.0 for c in rnd.choices(list(range(256)), k=3)])
    colors_map[0] = [1, 0, 0]
    colors_map[1] = [0, 1, 0]
    colors_map[2] = [0, 0, 1]

    for env_i, (point_dict, extras) in enumerate(render_outputs):
        if "all_coord" in extras:
            all_points = extras["all_coord"]  # assume debug mode
            all_colors = extras["all_color"]
            full_pcd = open3d.geometry.PointCloud()
            full_pcd.points = open3d.utility.Vector3dVector(all_points)
            full_pcd.colors = open3d.utility.Vector3dVector(all_colors)
        else:
            full_pcd = None

        points = point_dict["coord"]
        # colors = point_dict["color"] # Not used, colors generated from colors_map using class_idx
        if "class_idx" in point_dict:
            pts_types = point_dict["class_idx"]
        else:
            warnings.warn(
                "Missing class_idx in point_dict, will use ancestor_body_id instead", stacklevel=2
            )
            pts_types = point_dict["ancestor_body_id"]

        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        if isinstance(pts_types, torch.Tensor):
            pts_types = pts_types.cpu().numpy()

        colors = np.array([colors_map[class_idx] for class_idx in pts_types])

        if static:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = open3d.utility.Vector3dVector(colors)

            if full_pcd is not None:
                full_pcd.paint_uniform_color([1, 0, 0])

            all_geometries = [pcd] if full_pcd is None else [full_pcd, pcd]
            open3d.visualization.draw_geometries(all_geometries)
        elif env_i == 0:
            if o3d_vis is None:
                o3d_vis = open3d.visualization.Visualizer()
                o3d_vis.create_window()

            o3d_vis.clear_geometries()
            pcd0 = open3d.geometry.PointCloud()
            pcd0.points = open3d.utility.Vector3dVector(points)
            geometry = [pcd0]
            for g in geometry:
                o3d_vis.add_geometry(g)

            geometry[0].points = open3d.utility.Vector3dVector(points)
            geometry[0].colors = open3d.utility.Vector3dVector(colors)
            o3d_vis.update_geometry(geometry[0])
            # ctr = o3d_vis.get_view_control()
            # ctr.set_front([-0.6, 0.0, -0.8])
            # ctr.set_up([-0.8, 0.0, 0.6])
            o3d_vis.poll_events()
            o3d_vis.update_renderer()

            break

    return o3d_vis


def display_colored_pointclouds_interactive(
    render_outputs: list[PointsDict | ExtraDict | tuple[PointsDict, ExtraDict]],
    o3d_vis: open3d.visualization.Visualizer | None = None,
    geoms: list[open3d.geometry.PointCloud] | None = None,
) -> tuple[open3d.visualization.Visualizer | None, list[open3d.geometry.PointCloud] | None]:
    """static visualizations of obs from all threads"""
    if torch.cuda.is_available():
        log.warning("display() called with CUDA available. Ignoring call.")
        return None, None

    render_outputs = copy.deepcopy(render_outputs)

    for render_out in render_outputs:
        if isinstance(render_out, tuple):
            points_dict = {**render_out[0], **render_out[1]}
        elif isinstance(render_out, dict):
            points_dict = render_out
        else:
            raise NotImplementedError

        all_points = points_dict.get("all_coord", None)  # assume debug mode
        # assert all_points is not None
        if all_points is not None:
            full_pcd = open3d.geometry.PointCloud()
            full_pcd.points = open3d.utility.Vector3dVector(all_points)
            all_colors: np.ndarray = points_dict.get("all_color", None)
            if all_colors is not None:
                alpha = 0.25
                full_pcd.colors = open3d.utility.Vector3dVector(
                    alpha * all_colors + (1 - alpha) * np.array([1, 0.7, 0.7])
                )
            else:
                full_pcd.paint_uniform_color([1, 0.7, 0.7])

            if o3d_vis is None:
                o3d_vis = open3d.visualization.Visualizer()
                o3d_vis.create_window()
                geoms = [full_pcd]
                o3d_vis.add_geometry(geoms[0])
            else:
                geoms[0].points = full_pcd.points
                geoms[0].colors = full_pcd.colors
                o3d_vis.update_geometry(geoms[0])

        points = points_dict["coord"]
        colors = points_dict.get("color", None)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = open3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.7, 1, 0.7])

        if len(geoms) == 1:
            geoms.append(pcd)
            o3d_vis.add_geometry(geoms[1])
        else:
            geoms[1].points = pcd.points
            geoms[1].colors = pcd.colors
            o3d_vis.update_geometry(geoms[1])

        o3d_vis.poll_events()
        o3d_vis.update_renderer()

    return o3d_vis, geoms


class InteractiveVisualizer:
    def __init__(self) -> None:
        self.visualizer: open3d.visualization.Visualizer | None = None
        self.geoms = []
        self.ngeom = 0

    def visualize(
        self, geoms: list[open3d.geometry.PointCloud] | open3d.geometry.PointCloud
    ) -> None:
        if not isinstance(geoms, list):
            geoms = [geoms]
        if self.visualizer is None:
            self.visualizer = open3d.visualization.Visualizer()
            self.visualizer.create_window()
            self.ngeom = len(geoms)
            self.geoms = geoms

            for i in range(self.ngeom):
                self.visualizer.add_geometry(self.geoms[i])

        else:
            assert len(geoms) == self.ngeom
            for i in range(self.ngeom):
                if isinstance(geoms[i], open3d.geometry.PointCloud):
                    self.geoms[i].points = geoms[i].points
                    self.geoms[i].colors = geoms[i].colors
                elif isinstance(geoms[i], open3d.geometry.OrientedBoundingBox):
                    self.geoms[i].center = geoms[i].center
                    self.geoms[i].R = geoms[i].R
                    self.geoms[i].extent = geoms[i].extent
                else:
                    raise NotImplementedError(geoms[i])

                self.visualizer.update_geometry(self.geoms[i])

            self.visualizer.poll_events()
            self.visualizer.update_renderer()

    def reset(self) -> None:
        self.visualizer = None
        self.geoms = []
        self.ngeom = 0
