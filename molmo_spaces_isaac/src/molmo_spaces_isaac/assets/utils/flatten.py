import shutil
from pathlib import Path

from pxr import Sdf, Usd, UsdLux, UsdShade

from .data import BaseConversionData, Tokens


def export_flatten(
    data: BaseConversionData, usd_path: Path, format: str, is_scene: bool = False
) -> None:
    stage = data.stage
    f_name = usd_path.parent / f"{usd_path.stem}_flattened.{format}"
    flattened_layer = Sdf.Layer.CreateNew(f_name.absolute().as_posix())
    flattened_layer.ImportFromString(stage.Flatten().ExportToString())
    flattened_layer.Save()

    temp_textures_dir = usd_path.parent / Tokens.PAYLOAD.value / Tokens.TEXTURES.value
    output_textures_dir = usd_path.parent / Tokens.TEXTURES.value

    textures_to_copy: dict[Path, Path] = {}

    f_stage = Usd.Stage.Open(f_name.absolute().as_posix())
    for prim in f_stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            file_input = shader.GetInput("file")
            if file_input and file_input.Get() is not None:
                file_path = Path(
                    file_input.Get().path if hasattr(file_input.Get(), "path") else file_input.Get()
                )
                new_path = f"./{Tokens.TEXTURES.value}/{file_path.name}"
                file_input.Set(Sdf.AssetPath(new_path))
                if is_scene:
                    textures_to_copy[file_path] = output_textures_dir / file_path.name
        elif prim.IsA(UsdLux.DomeLight):
            light = UsdLux.DomeLight(prim)
            file_input = light.GetTextureFileAttr()
            if file_input and file_input.Get() is not None:
                file_path = Path(
                    file_input.Get().path if hasattr(file_input.Get(), "path") else file_input.Get()
                )
                new_path = f"./{Tokens.TEXTURES.value}/{file_path.name}"
                file_input.Set(Sdf.AssetPath(new_path))
                if is_scene:
                    textures_to_copy[file_path] = output_textures_dir / file_path.name

    f_stage.Save()

    if temp_textures_dir.is_dir():
        if output_textures_dir.exists():
            shutil.rmtree(output_textures_dir)
        shutil.copytree(temp_textures_dir, output_textures_dir, dirs_exist_ok=True)

    for src_path, dst_path in textures_to_copy.items():
        if src_path.is_file():
            shutil.copyfile(src_path, dst_path)

    shutil.rmtree(usd_path.parent / Tokens.PAYLOAD.value)

    if usd_path.is_file():
        usd_path.unlink()
    f_name.rename(usd_path)
