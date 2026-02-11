from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from .manager import KNOWN_URLS, ResourceManager

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
