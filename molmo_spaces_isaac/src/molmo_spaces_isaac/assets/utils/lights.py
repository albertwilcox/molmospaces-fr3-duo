from pxr import UsdGeom, UsdLux

from molmo_spaces_isaac.assets.utils.data import BaseConversionData, Tokens

DEFAULT_DIR_LIGHT_INTENSITY = 1000
DEFAULT_DIR_LIGHT_ROTATION_X = -10.0


def create_lights(data: BaseConversionData) -> None:
    # NOTE(wilbert): for now, just create a default directional light that kind of look fine
    content_stage = data.content[Tokens.CONTENTS]
    dir_light = UsdLux.DistantLight.Define(
        content_stage, content_stage.GetDefaultPrim().GetPath().AppendChild("scene_dir_light")
    )
    dir_light.GetIntensityAttr().Set(DEFAULT_DIR_LIGHT_INTENSITY)
    dir_light_prim = dir_light.GetPrim()
    dir_light_xform = UsdGeom.Xformable(dir_light_prim)
    dir_light_xform.AddRotateXOp().Set(DEFAULT_DIR_LIGHT_ROTATION_X)
