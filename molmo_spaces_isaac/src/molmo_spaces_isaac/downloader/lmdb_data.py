import json
import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lmdb
import zstandard
from tqdm import tqdm

MAP_SIZE = 10_000_000_000  # 10 GiB

COMPLETE_KEY = "__LMDBMap__dump_complete__"


class GenericLMDBMap(ABC):
    _ENCODED_COMPLETE_KEY = COMPLETE_KEY.encode("utf-8")

    def __init__(self, data_dir: str | Path, readonly=True, map_size=MAP_SIZE):
        """
        Args:
            data_dir (str): Path to LMDB database directory.
            readonly (bool): Open the database in readonly mode (for workers).
            map_size (int): Max size in bytes (needed only for writer mode).
        """
        self.pid = None
        self.db_path = str(data_dir)
        self.readonly = readonly
        self.map_size = map_size
        self.env = None

    def _open_env(self):
        self.env = lmdb.open(
            self.db_path,
            readonly=self.readonly,
            map_size=self.map_size,
            max_readers=2048,
            lock=not self.readonly,  # No lock needed for read-only
            readahead=True,
            meminit=False,
        )

    def _check_pid(self):
        if os.getpid() != self.pid:
            # Fork detected: close and reopen
            self.close()
            self.pid = os.getpid()
            self._open_env()

    @classmethod
    def database_exists(cls, data_dir: str | Path, map_size=MAP_SIZE):
        if not os.path.exists(data_dir):
            return False

        could_open = False
        db = cls(data_dir, map_size=map_size)
        try:
            db._check_pid()
            could_open = True

            complete_key = db.get(COMPLETE_KEY)

            if complete_key is None:
                return False
            if complete_key != COMPLETE_KEY:
                raise ValueError(
                    f"database complete key {complete_key} differs from expected {COMPLETE_KEY}"
                )

            return True
        except lmdb.Error:
            return False
        finally:
            if could_open:
                db.close()

    @classmethod
    @abstractmethod
    def serialize(cls, datum: Any) -> bytes:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized_datum: bytes) -> Any:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, str_to_serializable_map, data_dir, map_size=MAP_SIZE):
        def put(k, v):
            txn.put(k.encode("utf-8"), zstandard.compress(cls.serialize(v)))

        os.makedirs(data_dir, exist_ok=True)
        instance = cls(data_dir, readonly=False, map_size=map_size)
        instance._check_pid()

        with instance.env.begin(write=True) as txn:
            for key, value in tqdm(str_to_serializable_map.items(), "LMDB"):
                put(key, value)
            # Add completion key
            put(COMPLETE_KEY, COMPLETE_KEY)
            instance.env.sync()

        instance.close()

        return cls(data_dir, map_size=map_size)

    def get(self, key, alt=None):
        self._check_pid()
        with self.env.begin() as txn:
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                return alt
            return self.deserialize(zstandard.decompress(raw))

    def __iter__(self):
        yield from self.keys()

    def __contains__(self, key):
        return self.get(key) is not None

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        self._check_pid()
        with self.env.begin() as txn:
            stats = txn.stat()
            return stats["entries"] - 1  # ignore complete key

    def iterator(self, fn: Callable[[bytes, bytes], Any]):
        self._check_pid()
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for k, v in cursor:
                    if k != self._ENCODED_COMPLETE_KEY:
                        yield fn(k, v)

    def keys(self):
        yield from self.iterator(lambda k, v: k.decode("utf-8"))

    def values(self):
        yield from self.iterator(lambda k, v: self.deserialize(zstandard.decompress(v)))

    def items(self):
        yield from self.iterator(
            lambda k, v: (k.decode("utf-8"), self.deserialize(zstandard.decompress(v)))
        )

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = None
        self.pid = None


class JSonLMDBMap(GenericLMDBMap):
    @classmethod
    def serialize(cls, datum: Any) -> bytes:
        return json.dumps(datum).encode("utf-8")

    @classmethod
    def deserialize(cls, serialized_datum: bytes) -> Any:
        return json.loads(serialized_datum.decode("utf-8"))


class PickleLMDBMap(GenericLMDBMap):
    @classmethod
    def serialize(cls, datum: Any) -> bytes:
        return pickle.dumps(datum)

    @classmethod
    def deserialize(cls, serialized_datum: bytes) -> Any:
        return pickle.loads(serialized_datum)
