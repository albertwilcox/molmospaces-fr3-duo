from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from pxr import Sdf

from .manager import KNOWN_URLS, ResourceManager, SourceInfo

DEFAULT_CACHE_DIR = Path.home() / ".molmospaces"

SOURCE_TO_VERSION = {
    "objects": {
        "mjcf": {
            "thor": "20251117",
            "objaverse": "20251016_from_20250610",
        },
        "usd": {
            "thor": "20260128",
            "objaverse": "20260128",
        },
    },
    "scenes": {
        "mjcf": {
            "ithor": "20251217",
            "procthor-10k-train": "20251122",
            "procthor-10k-val": "20251217",
            "procthor-10k-test": "20251121",
            "holodeck-objaverse-train": "20251217",
            "holodeck-objaverse-val": "20251217",
            "procthor-objaverse-train": "20251205",
            "procthor-objaverse-val": "20251205",
        },
        "usd": {
            "ithor": "20260121",
            "procthor-10k-train": "20260128",
            "procthor-10k-val": "20260128",
            "procthor-10k-test": "20260128",
            "procthor-objaverse-train": "20260128",
            "procthor-objaverse-val": "20260128",
            "holodeck-objaverse-train": "20260128",
            "holodeck-objaverse-val": "20260128",
        },
    },
}

TYPE_TO_URL: dict[str, str] = {
    "mjcf": KNOWN_URLS["mujoco-thor-resources"],
    "usd": KNOWN_URLS["isaac-thor-resources"],
}


def archive_resolver(
    source: str, identifier: str, fn: Callable[[str, str, str], set[str]], data_type: str = "scenes"
) -> set[str]:
    archives = fn(source, identifier, data_type)

    if len(archives) == 0:
        raise ValueError(f"Unable to find archive for {source=} with {identifier=}")
    elif len(archives) > 1:
        raise RuntimeError(
            f"Bug in function to search for archives with numbers ({len(archives)} values returned)."
        )

    return archives


def archives_for_source_index(
    manager: ResourceManager, source: str, idx: int, data_type: str = "scenes"
) -> set[str]:
    fn = manager.archives_with_number
    return archive_resolver(source, str(idx), fn, data_type)


def install_scene_from_source_index(
    manager: ResourceManager, source: str, idx: int
) -> dict[str, set[str]]:
    source_to_archives = {source: archives_for_source_index(manager, source, idx)}
    manager.install_scenes(source_to_archives)
    return source_to_archives


def scene_path_resolve(
    manager: ResourceManager,
    source: str,
    idx: int,
    source_to_archives: dict[str, Collection[str]],
    variant: str = "",
) -> Path:
    if "procthor" in source or "holodeck" in source:
        fn = procthor_resolver
    elif "ithor" in source:
        fn = ithor_resolver
    else:
        raise NotImplementedError(f"Missing implementation for {source}")

    archive = list(source_to_archives[source])[0]
    scene_info = manager.scene_root_and_archive_paths(source)
    modalities = scene_info["archive_to_relative_paths"][archive]
    return fn(source, idx, scene_info, modalities, variant)


def ithor_resolver(
    source: str, idx: int, scene_info: SourceInfo, modalities: list[Path], variant: str = ""
) -> Path:
    return scene_info["root_dir"] / modalities[0]


def procthor_resolver(
    source: str, idx: int, scene_info: SourceInfo, modalities: list[Path], variant: str = ""
) -> Path:
    split = source.rsplit("-", maxsplit=1)[-1]
    target = f"{split}_{idx}" + variant
    assert Path(target) in modalities, f"Missing {target=} with available modalities {modalities}"
    return scene_info["root_dir"] / target


def find_object_paths(
    manager: ResourceManager,
    source: str,
    idx: int,
    source_to_archives: dict[str, set[str]],
    variant: str = "",
    exclude_thor=True,
) -> Iterator[tuple[str, Path]]:
    scene_path = scene_path_resolve(manager, source, idx, source_to_archives, variant)

    layer = Sdf.Layer.FindOrOpen(str(scene_path / "Payload" / "Geometry.usda"))

    external_assets = sorted(set(layer.externalReferences))
    if exclude_thor:
        external_assets = [asset for asset in external_assets if "/objects/thor/" not in asset]

    external_assets = [asset for asset in external_assets if "../objects/" in asset]

    for file_path in external_assets:
        # Objects are globally linked from the cache
        full_path = (scene_path / "Payload" / file_path).resolve()
        source = (full_path.relative_to(manager.cache_dir / "objects")).parts[0]
        rel_asset = full_path.relative_to(manager.object_dirs[source])
        yield source, rel_asset


def install_objects_for_scene(
    manager: ResourceManager,
    source_str: str,
    idx: int,
    scene_source_to_archive: dict[str, set[str]],
    variant: str = "",
    exclude_thor: bool = True,
) -> dict[str, list[str]]:
    source_to_archives = {}

    # TODO: check source_str, there was a name clashing here, so just patch it, but have to test as well
    for source, rel_asset in find_object_paths(
        manager, source_str, idx, scene_source_to_archive, variant, exclude_thor=exclude_thor
    ):
        archives = manager.archives_for_paths(source, [rel_asset], data_type="objects")
        if source not in source_to_archives:
            source_to_archives[source] = archives
        else:
            source_to_archives[source].extend(archives)

    source_to_archives: dict[str, list[str]] = {
        source: list(set(archives)) for source, archives in source_to_archives.items()
    }

    manager.install_objects(source_to_archives)

    return source_to_archives


def install_scene_with_objects(
    manager: ResourceManager, source: str, idx: int, variant: str = "", exclude_thor: bool = True
) -> dict[str, dict[str, Collection[str]]]:
    scene_source_to_archive = install_scene_from_source_index(manager, source, idx)
    objects_source_to_archives = install_objects_for_scene(
        manager, source, idx, scene_source_to_archive, variant, exclude_thor
    )
    return {"scenes": scene_source_to_archive, "objects": objects_source_to_archives}  # type: ignore


@dataclass
class DownloadArgs:
    type: Literal["mjcf", "usd"]

    assets: list[Literal["thor", "objaverse"]] = field(default_factory=lambda: ["thor"])

    scenes: list[
        Literal[
            "ithor",
            "procthor-10k-train",
            "procthor-10k-val",
            "procthor-10k-test",
            "procthor-objaverse-train",
            "procthor-objaverse-val",
            "holodeck-objaverse-train",
            "holodeck-objaverse-val",
        ]
    ] = field(default_factory=list)

    install_dir: Path | None = None

    cache_dir: Path = DEFAULT_CACHE_DIR

    house_index: int = -1

    clean: bool = False


def main() -> int:
    args = tyro.cli(DownloadArgs)

    if args.install_dir is None:
        print("Must provide an installation directory via --install-dir")
        return 1

    if not args.install_dir.is_dir():
        args.install_dir.mkdir(parents=True, exist_ok=True)

    assert args.type in TYPE_TO_URL, (
        f"Something went wrong, must only use 'mjcf' or 'usd', but got '{args.type}'"
    )

    print(f"[INFO]: saving to directory '{args.install_dir}'")
    print(f"[INFO]: downloading '{args.type}' version of the assets")

    sources_to_version = dict(objects=dict(), scenes=dict())
    sources_to_version["objects"]["thor"] = SOURCE_TO_VERSION["objects"][args.type]["thor"]
    for dataset_id in args.assets:
        sources_to_version["objects"][dataset_id] = SOURCE_TO_VERSION["objects"][args.type][
            dataset_id
        ]

    for dataset_id in args.scenes:
        sources_to_version["scenes"][dataset_id] = SOURCE_TO_VERSION["scenes"][args.type][
            dataset_id
        ]

    cache_dir = args.cache_dir / args.type
    cache_lock_file = cache_dir / ".lock"
    if cache_lock_file.is_file():
        cache_lock_file.unlink()

    install_lock_file = args.install_dir / ".lock"
    if install_lock_file.is_file():
        install_lock_file.unlink()

    manager = ResourceManager(
        base_url=TYPE_TO_URL[args.type],
        data_type_to_source_to_version=sources_to_version,
        symlink_dir=args.install_dir,
        cache_dir=args.cache_dir / args.type,
        force_install=True,
    )
    enforce_scenes = len(args.scenes) > 0
    enforce_objects = len(args.assets) > 1

    manager.install_sources(
        enforce_all_objects=enforce_objects,
        enforce_all_scenes=enforce_scenes,
        extract_only_scenes=False,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
