from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import msgspec


@dataclass
class BaseIsaaclabArgs:
    headless: bool = False
    livestream: int = -1
    enable_cameras: bool = False
    xr: bool = False
    device: str = "cuda:0"
    verbose: bool = False
    info: bool = False
    experience: str = ""
    rendering_mode: Literal["performance", "balanced", "quality"] = "quality"
    kit_args: str = ""
    anim_recording_enabled: bool = False
    anim_recording_start_time: float = 0
    anim_recording_stop_time: float = 10


class AssetGenMetadata(msgspec.Struct):
    asset_id: str
    hash_id: str
    articulated: bool
    bbox_size: list[float]


def load_thor_assets_metadata(filepath: Path) -> dict[str, AssetGenMetadata]:
    if not filepath.is_file():
        raise RuntimeError(f"The given file '{filepath.as_posix()}' is not a valid file")

    usd_assets_metadata: dict[str, AssetGenMetadata] = {}
    with open(filepath, "rb") as fhandle:
        usd_assets_metadata = msgspec.json.decode(fhandle.read(), type=dict[str, AssetGenMetadata])

    return usd_assets_metadata
