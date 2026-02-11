import contextlib
import gzip
import io
import json
import os
import re
import shutil
import stat
import tarfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from queue import Queue
from typing import TypedDict

import requests
import zstandard as zstd
from filelock import FileLock
from tqdm import tqdm

from molmo_spaces.resources.compact_trie import CompactPathTrie
from molmo_spaces.utils.lmdb_data import PickleLMDBMap

# R2 bucket public development access URL
KNOWN_URLS = {
    "mujoco-thor-resources": "https://pub-3555e9bb2d304fab9c6c79819e48aa40.r2.dev",
    "isaac-thor-resources": "https://pub-96496c3574b24d0c98b235219711d359.r2.dev",
}

TQDM_DISABLE_THRES = 3


REMOTE_MANIFEST_NAME = "mjthor_resource_file_to_size_mb.json"
LOCAL_MANIFEST_NAME = "mjthor_data_type_to_source_to_versions.json"
COMBINED_TRIES_NAME = "mjthor_resources_combined_meta.json.gz"


class SourceInfo(TypedDict):
    root_dir: Path
    archive_to_relative_paths: dict[str, list[Path]]


@contextlib.contextmanager
def lock_context(install_dir: Path | str | None, cache_dir: Path | str | None):
    """
    Locks local install dir, global cache dir, both, or neither, depending
    on whether the corresponding paths are given or set to None.

    If cache is known to be pre-installed, we can safely lock just the
    install dir
    """
    if install_dir is not None and cache_dir is not None:
        with (
            FileLock(os.path.join(install_dir, ".lock")),
            FileLock(os.path.join(cache_dir, ".lock")),
        ):
            yield
    elif install_dir is not None:
        with FileLock(os.path.join(install_dir, ".lock")):
            yield
    elif cache_dir is not None:
        with FileLock(os.path.join(cache_dir, ".lock")):
            yield
    else:
        yield


def safe_extract(
    tar,
    path: Path,
    read_only=True,
):
    resolve_root = path.resolve()

    for member in tar:
        member_path = path / member.name

        # Resolve path and check that it's inside the extract directory
        if not member_path.resolve().is_relative_to(resolve_root):
            raise Exception(f"Attempted path traversal in tar file: {member.name}")

        try:
            tar.extract(member, path=path)
        except PermissionError:
            if not member_path.exists():
                raise
            # Assume this was an extraction retry
            pass

        if read_only and member_path.is_file():
            # Set to read-only (user, group, others)
            member_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def complete_extract_flag_path(package: str, cache_destination: Path | str):
    return Path(cache_destination) / f".{package.replace('/', '__')}_complete_extract"


def complete_symlink_flag_path(package: str, symlink_destination: Path | str):
    return Path(symlink_destination) / f".{package.replace('/', '__')}_complete_links"


def download_and_extract_package(package, relative_path, cache_destination, read_only, base_url):
    try:
        if not package.endswith(".tar.zst"):
            raise Exception(f"Skipping {package} due to unknown extension")

        stream = requests.get(f"{base_url}/{relative_path.as_posix()}/{package}", stream=True)
        stream.raise_for_status()

        with zstd.ZstdDecompressor().stream_reader(stream.raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                safe_extract(tar, Path(cache_destination), read_only=read_only)

        # If we're here we could extract this package, so we can mark completion
        complete_flag = complete_extract_flag_path(package, cache_destination)
        complete_flag.touch(exist_ok=False)

        return True
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Download/extract failure for {package}: {e.__class__.name}: {e}")
        return False


def extract_worker(queue_in, queue_out):
    try:
        while True:
            work_data = queue_in.get()
            if work_data is None:
                break

            res = download_and_extract_package(*work_data)
            queue_out.put((work_data[0], res))
    finally:
        queue_out.put(None)


def logging_worker(queue_out, remote_manifest, num_workers, result_dict):
    pending_items = set(remote_manifest.keys())

    failed_items = set()

    num_ended = 0

    pbar = tqdm(
        total=len(remote_manifest),
        desc="Extracting",
        disable=len(remote_manifest) < TQDM_DISABLE_THRES,
    )

    try:
        while True:
            ret_val = queue_out.get()
            if ret_val is None:
                num_ended += 1
                if num_ended == num_workers:
                    break
                continue

            pending_items.remove(ret_val[0])
            if not ret_val[1]:
                failed_items.add(ret_val[0])

            pbar.update(1)
    finally:
        pbar.close()

    result_dict["failed_items"] = pending_items | failed_items
    result_dict["num_ended"] = num_ended


class ResourceManager:
    def __init__(
        self,
        base_url: str,
        data_type_to_source_to_version: dict,
        symlink_dir: Path,
        cache_dir: Path,
        force_install: bool = False,
        max_priority_returns: int = 1,
        cache_lock: bool = True,  # set to False if completely sure all data in use are pre-cached
        install_lock: bool = True,  # generally, don't touch
    ) -> None:
        assert Path(cache_dir).resolve() != Path(symlink_dir).resolve(), (
            f"symlink dir cannot be the same as the cache dir {Path(cache_dir).resolve()}"
        )

        print(
            f"resource manager ({base_url}) using:\n  {symlink_dir=}\n  {cache_dir=}\n  {force_install=}"
        )

        if not base_url.startswith("http"):
            assert base_url in KNOWN_URLS, (
                f"Please provide full URL for buckets other than {sorted(KNOWN_URLS.keys())}"
            )
            base_url = KNOWN_URLS[base_url]

        self.base_url = base_url

        if not cache_lock:
            print("Warning: skipping cache dir lock check")
        if not install_lock:
            print("Warning: skipping install dir lock check")

        self.data_type_to_source_to_version = data_type_to_source_to_version
        self.symlink_dir = symlink_dir
        self.cache_dir = cache_dir
        self.force_install = force_install
        self.max_priority_returns = max_priority_returns

        self.scene_dirs = {
            source: symlink_dir / "scenes" / source
            for source in data_type_to_source_to_version.get("scenes", {})
        }

        self.object_dirs = {
            source: cache_dir / "objects" / source / version
            for source, version in data_type_to_source_to_version.get("objects", {}).items()
        }

        self.grasp_dirs = {
            source: cache_dir / "grasps" / source / version
            for source, version in data_type_to_source_to_version.get("grasps", {}).items()
        }

        self._data_type_dirs = {
            "scenes": self.scene_dirs,
            "objects": self.object_dirs,
            "grasps": self.grasp_dirs,
        }

        # For scenes
        self._tries = {}
        self._number_to_archives = {}
        self._substring_to_scene_archives = {}

        # For objects
        self._object_tries = {}
        self._number_to_object_archives = {}
        self._substring_to_object_archives = {}

        # For grasps
        self._grasp_tries = {}
        self._number_to_grasp_archives = {}
        self._substring_to_grasp_archives = {}

        self.cache_lock = cache_lock
        self.install_lock = install_lock

    def install_sources(
        self,
        enforce_all_scenes=False,
        enforce_all_objects=False,
        enforce_all_grasps=False,
        extract_only_scenes=True,  # skips sym-linking
    ):
        self.ensure_installed_sources()

        if enforce_all_scenes:
            self.install_all_scenes(skip_linking=extract_only_scenes)

        if enforce_all_objects:
            self.install_all_objects()

        if enforce_all_grasps:
            self.install_all_grasps()

    def _install_items(
        self,
        data_type: str,
        source_to_install_packages: dict[str, list[str]],
        skip_linking: bool = False,
    ):
        source_to_version_to_install_packages = {}

        for source, install_packages in source_to_install_packages.items():
            source_to_version_to_install_packages[source] = {
                self.data_type_to_source_to_version[data_type][source]: install_packages
            }

        return self.ensure_installed_items(
            data_type=data_type,
            item_source_to_version_to_install_packages=source_to_version_to_install_packages,
            skip_scene_linking=skip_linking,
        )

    def find_all(self, data_type):
        to_install = {}
        for source in self.data_type_to_source_to_version[data_type]:
            to_install[source] = list(self.tries(source, data_type=data_type).keys())
            if len(to_install[source]) == 0:
                to_install.pop(source)
        return to_install

    def install_scenes(self, source_to_install_packages, skip_linking=False):
        return self._install_items("scenes", source_to_install_packages, skip_linking)

    def install_all_scenes(self, skip_linking=True) -> None:
        self.install_scenes(self.find_all("scenes"), skip_linking=skip_linking)

    def install_objects(self, source_to_install_packages):
        return self._install_items("objects", source_to_install_packages)

    def install_all_objects(self) -> None:
        self.install_objects(self.find_all("objects"))

    def install_grasps(self, source_to_install_packages):
        self._install_items("grasps", source_to_install_packages)

    def install_all_grasps(self) -> None:
        self.install_grasps(self.find_all("grasps"))

    def tries(self, source: str, data_type: str = "scenes") -> dict[str, CompactPathTrie]:
        if data_type == "scenes":
            self_tries = self._tries
            self_dirs = self.scene_dirs
        elif data_type == "objects":
            self_tries = self._object_tries
            self_dirs = self.object_dirs
        elif data_type == "grasps":
            self_tries = self._grasp_tries
            self_dirs = self.grasp_dirs
        else:
            raise NotImplementedError

        if source not in self_tries:
            try:
                # We save to a location under the install dir (should have fastest access)
                # rel_dir = self_dirs[source].relative_to(self.symlink_dir)
                base_dir = self.symlink_dir / ".lmdb"
                db_dir = base_dir / data_type / source

                if not PickleLMDBMap.database_exists(db_dir):
                    db_dir.mkdir(parents=True, exist_ok=True)
                    with lock_context(db_dir, None):
                        if not PickleLMDBMap.database_exists(db_dir):
                            print(f"Creating database for {source} ({data_type}) under {db_dir}")

                            with gzip.open(self_dirs[source] / COMBINED_TRIES_NAME, "rb") as f:
                                archive_to_paths = json.load(f)

                            archive_to_trie = {
                                archive: CompactPathTrie.from_dict(path_dict)
                                for archive, path_dict in archive_to_paths.items()
                            }

                            PickleLMDBMap.from_dict(archive_to_trie, db_dir)

                self_tries[source] = PickleLMDBMap(db_dir)
            except FileNotFoundError:
                self_tries[source] = {}

        return self_tries[source]

    def _data_type_root_and_archive_paths(
        self,
        data_type: str,
        source: str,
        mode: str = "recursive",  # or top_level
    ) -> SourceInfo:
        source_dir = self._data_type_dirs[data_type][source]

        if mode == "recursive":

            def fn(trie):
                return trie.all_paths()
        elif mode == "top_level":

            def fn(trie):
                return trie.root
        else:
            raise ValueError(f"Invalid mode {mode}")

        plain_paths = {
            archive: [Path(path) for path in fn(trie)]
            for archive, trie in self.tries(source, data_type=data_type).items()
        }

        return dict(root_dir=source_dir, archive_to_relative_paths=plain_paths)

    def scene_root_and_archive_paths_recursive(self, source: str) -> SourceInfo:
        return self._data_type_root_and_archive_paths("scenes", source, "recursive")

    def scene_root_and_archive_paths(self, source: str) -> SourceInfo:
        return self._data_type_root_and_archive_paths("scenes", source, "top_level")

    def object_root_and_archive_paths_recursive(self, source: str) -> SourceInfo:
        return self._data_type_root_and_archive_paths("objects", source, "recursive")

    def object_root_and_archive_paths(self, source: str) -> SourceInfo:
        return self._data_type_root_and_archive_paths("objects", source, "top_level")

    def grasp_root_and_archive_paths_recursive(self, source: str) -> SourceInfo:
        return self._data_type_root_and_archive_paths("grasps", source, "recursive")

    def grasp_root_and_archive_paths(self, source: str) -> SourceInfo:
        return self._data_type_root_and_archive_paths("grasps", source, "top_level")

    def archives_for_paths(
        self,
        data_source: str,
        paths: Sequence[Path],
        data_type: str = "scenes",
        use_numbers=None,  # internally defaults to True for scenes
        use_substrings=None,  # internally defaults to True for objects, grasps
    ):
        assert (use_numbers, use_substrings) not in [
            (True, True),
            (False, False),
        ], "use_numbers or use_substrings cannot be simultaneously True or False"

        source_dir = self._data_type_dirs[data_type][data_source]
        tries = self.tries(data_source, data_type=data_type)

        if data_type == "scenes":
            # For scenes: prefer numbers
            if use_numbers is None:
                use_numbers = not use_substrings
            if use_substrings is None:
                use_substrings = not use_numbers
        elif data_type in ["objects", "grasps"]:
            # For objects and grasps: prefer substrings
            if use_substrings is None:
                use_substrings = not use_numbers
            if use_numbers is None:
                use_numbers = not use_substrings
        else:
            raise NotImplementedError

        archives = set()

        for path in paths:
            if path.is_relative_to(source_dir):
                path = path.relative_to(source_dir)

            path = str(path)

            if use_numbers:
                numbers = self.extract_numbers(path)

                if numbers:
                    priority = set.union(
                        *[
                            self.archives_with_number(data_source, number, data_type)
                            for number in numbers
                            if len(self.archives_with_number(data_source, number, data_type))
                            <= self.max_priority_returns
                        ]
                    )
                else:
                    priority = set()
            elif use_substrings:
                substrings = self.extract_substrings(path)
                if substrings:
                    priority = set.union(
                        *[
                            self.archives_with_substring(data_source, substring, data_type)
                            for substring in substrings
                            if len(self.archives_with_substring(data_source, substring, data_type))
                            <= self.max_priority_returns
                        ]
                    )
                else:
                    # Is this even possible? Let's log
                    print(f"{path} has no substrings")
                    priority = set()
            else:
                raise ValueError("Bug: One of use_numbers and use_substrings must be True")

            for archive in chain(iter(priority), tries.keys()):
                if tries[archive].exists(path):
                    archives.add(archive)
                    break
            else:
                raise ValueError(f"No archive for {path}")

        return sorted(archives)

    def extract_numbers(self, text: str) -> Sequence[str]:
        return re.findall(r"-?\d+", text)

    def extract_substrings(self, text: str) -> Sequence[str]:
        # split by delimiters
        parts = re.split(r"[./_\s]+", text)
        return [p for p in parts if p]

    def archives_with_number(self, source: str, number: str, data_type: str = "scenes"):
        if data_type == "scenes":
            self_number_to_archives = self._number_to_archives
        elif data_type == "objects":
            self_number_to_archives = self._number_to_object_archives
        elif data_type == "grasps":
            self_number_to_archives = self._number_to_grasp_archives
        else:
            raise NotImplementedError

        if source not in self_number_to_archives:
            self_number_to_archives[source] = {}

            for archive in self.tries(source, data_type):
                numbers = set(self.extract_numbers(archive.replace(source, "")))
                for n in numbers:
                    if n in self_number_to_archives[source]:
                        self_number_to_archives[source][n].add(archive)
                    else:
                        self_number_to_archives[source][n] = {archive}

        return self_number_to_archives[source].get(number, set())

    def archives_with_substring(self, source: str, substring: str, data_type: str = "scenes"):
        if data_type == "scenes":
            self_substring_to_archives = self._substring_to_scene_archives
        elif data_type == "objects":
            self_substring_to_archives = self._substring_to_object_archives
        elif data_type == "grasps":
            self_substring_to_archives = self._substring_to_grasp_archives
        else:
            raise NotImplementedError

        if source not in self_substring_to_archives:
            res = {}
            for archive in self.tries(source, data_type):
                substrings = set(
                    self.extract_substrings(archive.replace(source, "").replace(".tar.zst", ""))
                )
                for s in substrings:
                    if s in res:
                        res[s].add(archive)
                    else:
                        res[s] = {archive}

            self_substring_to_archives[source] = res

        return self_substring_to_archives[source].get(substring, set())

    def archives_without_numbers(self, source: str, data_type: str = "scenes"):
        if data_type == "scenes":
            self_number_to_archives = self._number_to_archives
        elif data_type == "objects":
            self_number_to_archives = self._number_to_object_archives
        elif data_type == "grasps":
            self_number_to_archives = self._number_to_grasp_archives
        else:
            raise NotImplementedError

        self.archives_with_number(source, "0", data_type)
        all_archives = set(self.tries(source, data_type).keys())

        if self_number_to_archives[source]:
            all_numbered_archives = set.union(*list(self_number_to_archives[source].values()))
        else:
            all_numbered_archives = set()

        return sorted(all_archives - all_numbered_archives)

    def package_installer(
        self,
        packages,
        relative_path,
        symlink_or_dir,
        source: str,
        data_type: str,
        read_only=True,
        skip_scene_linking=False,
    ):
        # If cache lock is False, we assume everything is pre-extracted to cache

        # 1. Check if cache contains paths, else, download and extract
        if self.cache_lock:
            cache_destination = self.cache_dir / relative_path

            remote_manifest = self.cache_dir / relative_path / REMOTE_MANIFEST_NAME
            with open(remote_manifest, "r") as f:
                remote_manifest = json.load(f)

            download_manifest = {}
            for package in tqdm(
                packages,
                desc=f"Checking extracted {data_type}",
                disable=len(packages) < TQDM_DISABLE_THRES,
            ):
                assert package in remote_manifest

                complete_flag = complete_extract_flag_path(package, self.cache_dir / relative_path)
                if complete_flag.exists():
                    continue

                extracted_paths = self.tries(source, data_type)[package].leaf_paths()
                complete = True
                for epath in extracted_paths:
                    if not Path(self.cache_dir / relative_path / epath).exists():
                        complete = False
                        break
                if not complete:
                    download_manifest[package] = remote_manifest[package]
                else:
                    complete_flag.touch(exist_ok=False)

            if download_manifest:
                failures = self.extract_with_queue(
                    download_manifest,
                    relative_path,
                    cache_destination,
                    read_only,
                    data_type=data_type,
                    extract_package=True,
                )
                if failures:
                    raise Exception(
                        f"Failed to download {len(failures)} packages for {data_type} {source}:\n{failures}"
                    )

        if data_type in ["objects", "grasps"] or skip_scene_linking:
            # We just need objects and grasps to be extracted to the cache dir, since they use a global symlink
            return

        # 2. Check if install contains scene links, else, symlink
        for package in tqdm(
            packages, f"Linking {data_type}", disable=len(packages) < TQDM_DISABLE_THRES
        ):
            self.ensure_package_linked(
                symlink_or_dir=symlink_or_dir,
                package=package,
                trie=self.tries(source, data_type)[package],
                relative_path=relative_path,
            )

    def ensure_installed_items(
        self,
        data_type: str,
        item_source_to_version_to_install_packages: dict[str, dict[str, list[str]]] | None,
        skip_scene_linking: bool = False,
    ) -> None:
        target_dir = Path(self.symlink_dir).resolve()

        assert Path(target_dir).is_dir()
        assert Path(self.cache_dir).is_dir()

        # We still acquire the full lock to ensure nobody else is installing simultaneously
        with lock_context(
            target_dir if self.install_lock else None, self.cache_dir if self.cache_lock else None
        ):
            for (
                source,
                version_to_packages,
            ) in item_source_to_version_to_install_packages.items():
                for version, packages in version_to_packages.items():
                    print(
                        f"Ensuring {len(packages)} {data_type} archive(s) for {source} {version=}"
                    )
                    relative_path = Path(data_type) / source / version
                    symlink_or_dir = Path(target_dir) / data_type / source
                    self.package_installer(
                        packages,
                        relative_path,
                        symlink_or_dir,
                        read_only=True,
                        source=source,
                        data_type=data_type,
                        skip_scene_linking=skip_scene_linking,
                    )

    def default_installer(
        self,
        remote_manifest,
        relative_path,
        symlink_dir,
        data_type,
        read_only=True,
        must_download=False,
    ):
        cache_destination = self.cache_dir / relative_path

        if must_download:
            os.makedirs(cache_destination, exist_ok=True)

            if data_type in ["scenes"]:
                # lazy download all scenes
                enforce_extract = False
            elif data_type in ["grasps"]:
                if relative_path.parent.name in ["droid", "rum"]:
                    # thor-related grasps - always install if requested
                    enforce_extract = True
                else:
                    enforce_extract = False
            elif data_type in ["objects"]:
                # relative_path = Path(data_type) / source / version
                if relative_path.parent.name in ["thor", "objathor_metadata"]:
                    # thor objects and metadata - always install if requested
                    enforce_extract = True
                else:
                    enforce_extract = False
            else:  # e.g. robots, test_data, benchmarks
                enforce_extract = True

            msg = (
                "Downloading and extracting to cache"
                if enforce_extract
                else "Postponing downloading"
            )
            print(f"{msg} {sum(remote_manifest.values()):.1f} MB of compressed data...")

            failures = self.extract_with_queue(
                remote_manifest,
                relative_path,
                cache_destination,
                read_only,
                data_type,
                extract_package=enforce_extract,
            )
            if failures:
                raise Exception(
                    f"Failed to download {len(failures)} packages for {relative_path}:\n{failures}"
                )

        extracted_path = self.cache_dir / relative_path
        # Compute relative path from the link location to the target
        symlink_parent_dir = symlink_dir.parent
        symlink_parent_dir.mkdir(parents=True, exist_ok=True)

        if symlink_dir.is_symlink():
            symlink_dir.unlink()
        elif symlink_dir.is_dir():
            shutil.rmtree(symlink_dir)

        if data_type in ["grasps", "objects", "test_data", "benchmarks"]:
            print(f"Installing global symlink\n  from {extracted_path}\n  into {symlink_dir}")
            symlink_dir.symlink_to(extracted_path, target_is_directory=True)

        elif data_type in ["scenes", "robots"]:
            print(f"Creating empty dir\n  for {extracted_path}\n  into {symlink_dir}")
            symlink_dir.mkdir(parents=True, exist_ok=True)
            (symlink_dir / REMOTE_MANIFEST_NAME).symlink_to(extracted_path / REMOTE_MANIFEST_NAME)
            (symlink_dir / COMBINED_TRIES_NAME).symlink_to(extracted_path / COMBINED_TRIES_NAME)

        if data_type in ["robots"]:
            with gzip.open(symlink_dir / COMBINED_TRIES_NAME, "rb") as f:
                archive_to_paths = json.load(f)

            archive_to_trie = {
                archive: CompactPathTrie.from_dict(path_dict)
                for archive, path_dict in archive_to_paths.items()
            }

            for archive in tqdm(
                archive_to_trie,
                f"Linking {symlink_dir}",
                disable=len(archive_to_trie) < TQDM_DISABLE_THRES,
            ):
                self.ensure_package_linked(
                    symlink_or_dir=symlink_dir,
                    package=archive,
                    trie=archive_to_trie[archive],
                    relative_path=relative_path,
                )

    def ensure_installed_sources(self) -> None:
        def save_manifest(path, manifest):
            with open(path, "w") as mani_file:
                json.dump(manifest, mani_file, indent=2)

        def fetch_manifest(relative_path, manifest_name):
            remote_path = f"{relative_path}/{manifest_name}"
            print(f"Downloading to cache {remote_path}...")
            resp = requests.get(f"{self.base_url}/{remote_path}")
            resp.raise_for_status()

            with open(Path(cache_path) / manifest_name, "wb") as remote_dump_file:
                remote_dump_file.write(io.BytesIO(resp.content).read())

        target_dir = Path(self.symlink_dir).resolve()

        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        with lock_context(
            target_dir if self.install_lock else None,
            self.cache_dir if self.cache_lock else None,
        ):
            # This is for the central package install location
            local_manifest: dict[str, dict[str, list[str]]] = {}

            local_manifest_path = self.cache_dir / LOCAL_MANIFEST_NAME
            if os.path.exists(local_manifest_path):
                with open(local_manifest_path, "r") as f:
                    local_manifest.update(json.load(f))

            # This is for  the target location, which will be used
            target_manifest: dict[str, dict[str, str | None]] = {}

            target_manifest_path = target_dir / LOCAL_MANIFEST_NAME
            if os.path.exists(target_manifest_path):
                with open(target_manifest_path, "r") as f:
                    target_manifest.update(json.load(f))

            for data_type, source_to_version in self.data_type_to_source_to_version.items():
                if data_type not in local_manifest:
                    local_manifest[data_type] = {}

                for source, version in source_to_version.items():
                    if source not in local_manifest[data_type]:
                        local_manifest[data_type][source] = []

                    if data_type not in target_manifest:
                        target_manifest[data_type] = {}

                    if source not in target_manifest[data_type]:
                        target_manifest[data_type][source] = None
                    elif target_manifest[data_type][source] == version:
                        print(f"Skipping already installed {source} ({data_type}) with {version=}")
                        continue

                    print(f"Setting up {source} ({data_type}) with {version=}")

                    relative_path = Path(data_type) / source / version

                    symlink_or_dir = Path(target_dir) / data_type / source

                    if symlink_or_dir.is_symlink() and symlink_or_dir.readlink().name != version:
                        if not self.force_install:
                            raise ValueError(
                                f"Requested {version=} differs from previously installed one.\n"
                                f"You may want to delete\n  '{symlink_or_dir}' -> '{symlink_or_dir.readlink()}'\n"
                                f"to proceed.\n"
                                f"Alternatively, set `force_install=True` to overwrite pre-installed versions."
                            )
                        else:
                            symlink_or_dir.unlink()
                            # Also remove lmdb, if any
                            lmdb_path = target_dir / ".lmdb" / data_type / source
                            if lmdb_path.exists():
                                shutil.rmtree(lmdb_path)
                    elif symlink_or_dir.is_dir() and target_manifest[data_type][source] != version:
                        if not self.force_install:
                            raise ValueError(
                                f"Requested {version=} differs from previously installed one.\n"
                                f"You may want to delete\n  '{symlink_or_dir}'\n"
                                f"to proceed.\n"
                                f"Alternatively, set `force_install=True` to overwrite pre-installed versions."
                            )
                        else:
                            # shutil.rmtree(symlink)
                            # Remove lmdb, if any
                            # overwriting while keeping whatever is there
                            lmdb_path = target_dir / ".lmdb" / data_type / source
                            if lmdb_path.exists():
                                shutil.rmtree(lmdb_path)

                    cache_path = os.path.join(self.cache_dir, relative_path)

                    if version not in local_manifest[data_type][source]:
                        must_download = True

                        if os.path.exists(cache_path):
                            shutil.rmtree(cache_path)

                        Path(cache_path).mkdir(parents=True, exist_ok=True)

                        fetch_manifest(relative_path, REMOTE_MANIFEST_NAME)
                        fetch_manifest(relative_path, COMBINED_TRIES_NAME)

                    else:
                        must_download = False

                    with open(Path(cache_path) / REMOTE_MANIFEST_NAME, "r") as remote_dump_file:
                        remote_manifest = json.load(remote_dump_file)

                    try:
                        self.default_installer(
                            remote_manifest,
                            relative_path,
                            symlink_or_dir,
                            data_type=data_type,
                            read_only=True,
                            must_download=must_download,
                        )
                        if remote_manifest:
                            local_manifest[data_type][source].append(version)
                        target_manifest[data_type][source] = version
                    finally:
                        save_manifest(local_manifest_path, local_manifest)
                        save_manifest(target_manifest_path, target_manifest)

    def extract_with_queue(
        self,
        download_package_to_size,
        relative_path,
        cache_destination,
        read_only,
        data_type,
        max_num_workers=10,
        min_items_per_worker=10,
        extract_package: bool = False,
    ):
        if data_type in ["scenes", "objects", "grasps"] and not extract_package:
            return []

        queue_in = Queue()
        queue_out = Queue()

        for key in tqdm(
            list(download_package_to_size.keys()),
            "Queueing packages",
            disable=len(download_package_to_size) < TQDM_DISABLE_THRES,
        ):
            queue_in.put((key, relative_path, cache_destination, read_only, self.base_url))

        num_workers = max(
            min(max_num_workers, len(download_package_to_size) // min_items_per_worker), 1
        )

        for _ in range(num_workers):
            # Stops workers
            queue_in.put(None)

        if len(download_package_to_size) > 1:
            print(
                f"Downloading and extracting to cache {len(download_package_to_size)} packages. This can take a while..."
            )
        log_result = {}

        with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
            executor.submit(
                logging_worker, queue_out, download_package_to_size, num_workers, log_result
            )
            for _ in range(num_workers):
                executor.submit(extract_worker, queue_in, queue_out)

        failed_items = log_result["failed_items"]

        if not failed_items:
            return failed_items

        print(f"{failed_items=} ({len(failed_items)} entries)")

        if len(failed_items) > max(0.1 * len(download_package_to_size), 1):
            return failed_items

        # Fallback - retry download in a single thread

        print(
            f"Retrying {len(failed_items)} failed items in main thread (this may take a while)..."
        )

        retry_manifest = {item: download_package_to_size[item] for item in failed_items}

        queue_in = Queue()
        queue_out = Queue()

        for item in failed_items:
            queue_in.put((item, relative_path, cache_destination, read_only, self.base_url))
        queue_in.put(None)

        extract_worker(queue_in, queue_out)
        retry_result = {}
        logging_worker(queue_out, retry_manifest, num_workers=1, result_dict=retry_result)
        retry_failed_items = retry_result["failed_items"]

        return retry_failed_items

    def ensure_package_linked(
        self,
        symlink_or_dir: str | Path,
        package: str,
        trie: CompactPathTrie,
        relative_path: str | Path,
    ):
        complete_flag = complete_symlink_flag_path(package, symlink_or_dir)
        if complete_flag.exists():
            return

        extracted_paths = trie.leaf_paths()

        # First, create all directories
        dir_paths = CompactPathTrie.from_paths(trie.non_leaf_paths()).leaf_paths()
        for dir_path in dir_paths:
            full_dir_path = Path(symlink_or_dir / dir_path).resolve()
            os.makedirs(full_dir_path, exist_ok=True)

        # Then link all files from cache
        for epath in extracted_paths:
            if not Path(symlink_or_dir / epath).exists():
                target = (self.cache_dir / relative_path / epath).resolve()
                lnk = Path(symlink_or_dir / epath).resolve()
                os.symlink(target, lnk, target_is_directory=False)

        complete_flag.touch(exist_ok=False)
