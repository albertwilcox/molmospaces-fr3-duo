from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import sapien
import transforms3d as t3d
import tyro
from sapien.physx import PhysxArticulation

from molmo_spaces_maniskill.assets.loader import MjcfAssetActorLoader, MjcfAssetArticulationLoader


@dataclass
class Args:
    filepath: Path
    fixed_base: bool = False
    mode: Literal["articulation", "actor"] = "articulation"
    rotate: bool = False


FIX_ASSET_ROTATION = t3d.euler.euler2quat(np.pi / 2, 0, 0)


def show_articulation_tree(articulation: PhysxArticulation) -> None:
    link = articulation.get_root()
    stack = [link]
    depth = 0
    while len(stack) > 0:
        curr_link = stack.pop()
        joint = curr_link.get_joint()
        print(f"{'  ' * depth}body: {curr_link.name}")
        # print(f"{'  ' * depth}pose = {curr_link.get_entity_pose()}")
        if joint is not None:
            print(f"{'  ' * depth}joint: {joint.name}")
            print(f"{'  ' * depth}pose = {joint.get_pose_in_child()}")
            print(f"{'  ' * depth}pose = {joint.get_pose_in_parent()}")
        depth += 1
        for child in curr_link.get_children():
            stack.append(child)


def main() -> int:
    args = tyro.cli(Args)

    if not args.filepath.exists():
        print(f"Given mjcf model @ {args.filepath.as_posix()} doesn't exist")
        return 1

    scene = sapien.Scene()
    scene.set_timestep(0.002)

    scene.add_ground(altitude=0)

    entity: sapien.Entity | PhysxArticulation | None = None
    if args.mode == "articulation":
        loader = MjcfAssetArticulationLoader(scene)
    else:
        loader = MjcfAssetActorLoader(scene)
    builder = loader.load_from_xml(args.filepath, floating_base=not args.fixed_base)
    entity = builder.build()

    # Uncomment to print the tree structure of the loaded articulation
    # if isinstance(entity, PhysxArticulation):
    #     show_articulation_tree(entity)

    initial_pose = sapien.Pose(
        p=[0.0, 0.0, 1.25],
        q=(FIX_ASSET_ROTATION.tolist() if args.rotate else [1, 0, 0, 0]),
    )
    entity.set_pose(initial_pose)

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-12, y=0, z=15)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
    assert viewer.window is not None
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    # viewer.focus_entity(car.root.entity)
    if isinstance(entity, PhysxArticulation):
        viewer.focus_entity(entity.root.entity)
    else:
        viewer.focus_entity(entity)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)

    running = True
    while not viewer.closed:
        if viewer.window.key_down("r"):
            viewer.close()
            break
        if viewer.window.key_down("delete"):
            entity.set_pose(initial_pose)
            if isinstance(entity, PhysxArticulation):
                entity.set_root_linear_velocity([0, 0, 0])
                entity.set_root_angular_velocity([0, 0, 0])
        elif viewer.window.key_down("space"):
            running = not running

        if running:
            scene.step()
        scene.update_render()
        viewer.render()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
