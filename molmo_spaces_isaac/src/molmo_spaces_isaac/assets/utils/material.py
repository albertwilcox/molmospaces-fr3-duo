import shutil
from pathlib import Path
from typing import Any

import mujoco as mj
import usdex.core
from pxr import Gf, Sdf, Usd, UsdShade

from .data import BaseConversionData, Tokens


def convert_materials(data: BaseConversionData, prefix: str = "") -> None:
    if len(data.spec.materials) < 1:
        return

    material_lib = usdex.core.addAssetLibrary(
        data.content[Tokens.CONTENTS], Tokens.MATERIALS.value, format="usda"
    )
    assert material_lib is not None, (
        f"Couldn't create material library for model {data.spec.modelname}"
    )

    data.libraries[Tokens.MATERIALS] = material_lib
    data.references[Tokens.MATERIALS] = {}

    materials_scope = data.libraries[Tokens.MATERIALS].GetDefaultPrim()
    orig_names: list[str] = [f"{prefix}{mat.name}" for mat in data.spec.materials]
    safe_names: list[str] = data.name_cache.getPrimNames(materials_scope, orig_names)

    for material, safe_name in zip(data.spec.materials, safe_names):
        assert isinstance(material, mj.MjsMaterial)
        material_prim = convert_material(
            materials_scope, safe_name, material, data.spec, material_lib
        ).GetPrim()
        data.references[Tokens.MATERIALS][material.name] = material_prim

    usdex.core.saveStage(
        data.libraries[Tokens.MATERIALS],
        comment=f"Material library for {data.spec.modelname}. {data.comment}",
    )

    material_asset_content = usdex.core.addAssetContent(
        data.content[Tokens.CONTENTS], Tokens.MATERIALS.value, format="usda"
    )
    assert material_asset_content is not None, (
        f"Couldn't create asset content for material, for model {data.spec.modelname}"
    )

    data.content[Tokens.MATERIALS] = material_asset_content


def convert_material(
    parent: Usd.Prim,
    safe_name: str,
    material: mj.MjsMaterial,
    spec: mj.MjSpec,
    material_lib: Usd.Stage,
) -> UsdShade.Material:
    # TODO(wilbert): opacity seems to make some assets look weird (e.g. Fridge_19), so for now
    # we're disabling it until we find a solution to set the correct alpha blending mode
    color, _ = Gf.Vec3f(*material.rgba[:3].tolist()), material.rgba[-1]

    material_kwargs: dict[str, Any] = dict(color=color, opacity=1.0)
    if material.shininess != -1.0:
        material_kwargs["roughness"] = 1.0 - material.shininess
    if material.metallic != -1.0:
        material_kwargs["metallic"] = material.metallic

    material_prim = usdex.core.definePreviewMaterial(parent, safe_name, **material_kwargs)

    # TODO(wilbert): check if we should enable specular and emissive components. They look nicer,
    # but that's just my opinion

    # if material.specular != 0:
    #     surface_shader = usdex.core.computeEffectivePreviewSurfaceShader(material_prim)
    #     surface_shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    #     surface_shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
    #         Gf.Vec3f(material.specular)
    #     )

    if material.emission != spec.default.material.emission:
        surface_shader = usdex.core.computeEffectivePreviewSurfaceShader(material_prim)
        surface_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(material.emission)
        )

    if main_texture_name := material.textures[mj.mjtTextureRole.mjTEXROLE_RGB.value]:
        texture_handle = spec.texture(main_texture_name)
        if texture_handle is not None and not texture_handle.builtin:
            texture_path: Sdf.AssetPath = convert_texture(texture_handle, spec, material_lib)
            if texture_path and not usdex.core.addDiffuseTextureToPreviewMaterial(
                material_prim, texture_path
            ):
                print(
                    f"[WARN]: Failed to add diffuse texture for material prim {material_prim.GetPrim().GetPath()}"
                )
            else:
                shader_path = material_prim.GetPrim().GetPath().AppendPath("DiffuseTexture")
                shader_prim = parent.GetStage().GetPrimAtPath(shader_path)
                if shader_prim.IsValid() and shader_prim.IsA(UsdShade.Shader):
                    shader = UsdShade.Shader(shader_prim)
                    shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                    shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

                    shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                        Gf.Vec4f(color[0], color[1], color[2], 1.0)
                    )

    elif any(material.textures):
        print(f"[WARN]: Unsupported texture layers for material {safe_name}")

    return material_prim


def convert_texture(
    texture: mj.MjsTexture, spec: mj.MjSpec, material_lib: Usd.Stage
) -> Sdf.AssetPath:
    texture_path = Path(spec.modelfiledir) / spec.texturedir / texture.file
    if not texture_path.is_file():
        raise RuntimeError(f"Texture @ '{texture_path}' doesn't exist")

    local_texture_dir = Path(material_lib.GetRootLayer().identifier).parent / Tokens.TEXTURES.value
    if not local_texture_dir.is_dir():
        local_texture_dir.mkdir(exist_ok=True)
    local_texture_path = local_texture_dir / texture_path.name
    shutil.copyfile(src=texture_path, dst=local_texture_path)

    rel_texture_path = local_texture_path.relative_to(
        Path(material_lib.GetRootLayer().identifier).parent
    )
    return Sdf.AssetPath(f"./{rel_texture_path.as_posix()}")
