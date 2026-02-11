import shutil
from pathlib import Path

import mujoco as mj
import usdex.core
from pxr import UsdLux

from molmo_spaces_isaac.assets.utils.data import BaseConversionData, Tokens

DEFAULT_DOME_LIGHT_INTENSITY = 1000


def create_skybox(data: BaseConversionData) -> UsdLux.DomeLight | None:
    skybox_filepath: Path | None = None
    for texture in data.spec.textures:
        assert isinstance(texture, mj.MjsTexture)
        if texture.type == mj.mjtTexture.mjTEXTURE_SKYBOX:
            filepath = Path(data.spec.modelfiledir) / texture.file
            if filepath.is_file():
                skybox_filepath = filepath
                break

    if skybox_filepath is None:
        return None

    material_lib = data.libraries[Tokens.MATERIALS]
    local_texture_dir = Path(material_lib.GetRootLayer().identifier).parent / Tokens.TEXTURES.value
    if not local_texture_dir.is_dir():
        local_texture_dir.mkdir(exist_ok=True)

    local_skybox_filepath = local_texture_dir / skybox_filepath.name
    shutil.copyfile(src=skybox_filepath, dst=local_skybox_filepath)

    rel_skybox_filepath = local_skybox_filepath.relative_to(
        Path(material_lib.GetRootLayer().identifier).parent
    )

    dome_light = usdex.core.defineDomeLight(
        parent=data.content[Tokens.CONTENTS].GetDefaultPrim(),
        name="scene_skybox_light",
        intensity=1000,
        texturePath=f"./{rel_skybox_filepath.as_posix()}",
    )

    return dome_light
