import json
import os
from pathlib import Path

from molmo_spaces.resources.manager import ResourceManager

_RESOURCE_MANAGER = {}


def str2bool(v: str):
    v = v.lower().strip()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"{v} cannot be converted to a bool")


def prompt_yes_no(message, skip_prompt=False):
    if skip_prompt:
        return True
    r"""Prints a message and prompts the user for "y" or "n" returning True or False."""
    # https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/utils/datasets_download.py
    print("\n-------------------------")
    print(message)
    while True:
        answer = input("(y|n): ")
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            print("Invalid answer...")


AUTO_INSTALL = str2bool(os.environ.get("MLSPACES_AUTO_INSTALL", "True"))
FORCE_INSTALL = str2bool(os.environ.get("MLSPACES_FORCE_INSTALL", "True"))


def setup_resource_manager(
    BUCKET_OR_URL, ASSETS_DIR, DATA_TYPE_TO_SOURCE_TO_VERSION, DATA_CACHE_DIR
):
    key = ""
    for data_type in sorted(DATA_TYPE_TO_SOURCE_TO_VERSION.keys()):
        key += f"__{data_type}_"
        source_to_version = DATA_TYPE_TO_SOURCE_TO_VERSION[data_type]
        for source in sorted(source_to_version.keys()):
            key += f"_{source}_{source_to_version[source]}_"
    global _RESOURCE_MANAGER
    if key not in _RESOURCE_MANAGER:
        _RESOURCE_MANAGER[key] = ResourceManager(
            base_url=BUCKET_OR_URL,
            data_type_to_source_to_version=DATA_TYPE_TO_SOURCE_TO_VERSION,
            symlink_dir=ASSETS_DIR,
            cache_dir=DATA_CACHE_DIR,
            force_install=FORCE_INSTALL,
            cache_lock=str2bool(os.environ.get("MLSPACES_CACHE_LOCK", "True")),
        )

        current_manager = _RESOURCE_MANAGER[key]

        def _get_current_install():
            current_install = {}

            manifest_path = (
                Path(ASSETS_DIR) / "mlspaces_installed_data_type_to_source_to_versions.json"
            )

            if manifest_path.is_file():
                with open(manifest_path, "r") as f:
                    current_install.update(json.load(f))

            for data_type, source_to_version in DATA_TYPE_TO_SOURCE_TO_VERSION.items():
                if data_type not in current_install:
                    current_install[data_type] = {}
                for source in source_to_version:
                    if source not in current_install[data_type]:
                        current_install[data_type][source] = None

            return current_install

        _previous_install = _get_current_install()

        def _should_install():
            return any(
                _previous_install[data_type][source] != version
                for data_type, source_to_version in DATA_TYPE_TO_SOURCE_TO_VERSION.items()
                for source, version in source_to_version.items()
            )

        if _should_install() and prompt_yes_no(
            f"Install data versions\n{json.dumps(DATA_TYPE_TO_SOURCE_TO_VERSION, indent=2)}\nunder {ASSETS_DIR}?\n"
            f"The current install is\n{json.dumps(_previous_install, indent=2)}",
            skip_prompt=AUTO_INSTALL,
        ):
            print(
                f"Installing missing data versions. This may take a while... (if hanging delte {DATA_CACHE_DIR} and try again)"
            )

            if not os.environ.get("_IN_MULTIPROCESSING_CHILD") and str2bool(
                os.environ.get("MLSPACES_DOWNLOAD_EXTRACT_ALL_SCENES_OBJECTS_GRASPS", "False")
            ):
                current_manager.install_sources(
                    enforce_all_scenes=True, enforce_all_objects=True, enforce_all_grasps=True
                )
            else:
                current_manager.install_sources()

        def _install_scenes_default():
            to_install = {}
            for scene_source in DATA_TYPE_TO_SOURCE_TO_VERSION.get("scenes", {}):
                num_packages = len(current_manager.tries(scene_source))

                if num_packages < 10:
                    packages = list(current_manager.tries(scene_source).keys())
                else:
                    packages = current_manager.archives_without_numbers(scene_source)

                if packages:
                    to_install[scene_source] = packages

            if to_install:
                current_manager.install_scenes(to_install)

        # Only install scenes at import time if not in a multiprocessing child context
        # In distributed/worker contexts, scenes should already be installed
        if not os.environ.get("_IN_MULTIPROCESSING_CHILD"):
            _install_scenes_default()

    # assert ASSETS_DIR == _RESOURCE_MANAGER.symlink_dir
    # # assert DATA_TYPE_TO_SOURCE_TO_VERSION == _RESOURCE_MANAGER.data_type_to_source_to_version
    # assert DATA_CACHE_DIR == _RESOURCE_MANAGER.cache_dir

    return _RESOURCE_MANAGER[key]
