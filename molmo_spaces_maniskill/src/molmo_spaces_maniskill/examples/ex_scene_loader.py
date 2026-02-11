from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sapien
import tyro

from molmo_spaces_maniskill.assets.loader import MjcfSceneLoader


@dataclass
class Args:
    filepath: Path


def main() -> int:
    args = tyro.cli(Args)

    if not args.filepath.is_file():
        print(f"Given scene @ {args.filepath.as_posix()} doesn't exist")
        return 1

    sapien.render.set_camera_shader_dir("default")
    sapien.render.set_viewer_shader_dir("default")

    # sapien.render.set_camera_shader_dir("rt")
    # sapien.render.set_viewer_shader_dir("rt")
    # sapien.render.set_ray_tracing_samples_per_pixel(4)  # change to 256 for less noise
    # sapien.render.set_ray_tracing_denoiser("optix") # change to "optix" or "oidn"

    scene = sapien.Scene()
    scene.set_timestep(0.002)

    loader = MjcfSceneLoader(scene)
    actors, articulations = loader.load(args.filepath)

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-12, y=0, z=15)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
    assert viewer.window is not None
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    scene.set_ambient_light([0.5, 0.5, 0.5])

    while not viewer.closed:
        if viewer.window.key_down("q"):
            viewer.close()
            break

        scene.step()
        scene.update_render()
        viewer.render()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
