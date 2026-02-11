import json


class CompactPathTrie:
    def __init__(self, sep="/"):
        self.root = dict()
        self.sep = sep

    # Insert a path (no compaction)
    def insert(self, path: str):
        parts = path.strip(self.sep).split(self.sep)
        node = self.root
        for part in parts:
            if part not in node:
                node[part] = {}
            node = node[part]

    # Path existence check
    def exists(self, path: str) -> bool:
        parts = path.strip(self.sep).split(self.sep)
        node = self.root
        i = 0
        while i < len(parts):
            found = False
            for k in node:
                k_parts = k.split(self.sep)
                if parts[i : i + len(k_parts)] == k_parts:
                    node = node[k]
                    i += len(k_parts)
                    found = True
                    break
            if not found:
                return False
        return True

    # Get all paths
    def all_paths(self, include_intermediate=True):
        return self._collect_paths(self.root, prefix="", include_intermediate=include_intermediate)

    def leaf_paths(self):
        return self._collect_paths(self.root, prefix="", include_intermediate=False)

    def non_leaf_paths(self):
        return list(set(self.all_paths()) - set(self.leaf_paths()))

    def _collect_paths(self, node, prefix, include_intermediate=True):
        paths = []
        for k, v in node.items():
            new_prefix = f"{prefix}/{k}" if prefix else k
            if include_intermediate or not v:
                paths.append(new_prefix)
            if v or include_intermediate:
                paths.extend(self._collect_paths(v, new_prefix, include_intermediate))

        return paths

    # JSON-friendly
    def to_dict(self):
        return self.root

    def to_json(self, **kwargs):
        return json.dumps(self.root, **kwargs)

    # --- COMPACTION STEP ---
    def compact(self):
        self.root = self._compact_node(self.root)

    def _compact_node(self, node):
        if not node:
            return {}

        new_node = {}
        for k, v in node.items():
            # Recursively compact children first
            compacted_child = self._compact_node(v)

            # Accumulate keys along chains with only one child
            while len(compacted_child) == 1 and next(iter(compacted_child.values())):
                child_key, child_val = next(iter(compacted_child.items()))
                k = f"{k}{self.sep}{child_key}"
                compacted_child = child_val

            new_node[k] = compacted_child
        return new_node

    # --- Load from dict ---
    @classmethod
    def from_dict(cls, d, sep="/"):
        trie = cls(sep)
        trie.root = d
        trie.compact()
        return trie

    # --- Load from JSON string ---
    @classmethod
    def from_json(cls, s, sep="/"):
        d = json.loads(s)
        return cls.from_dict(d, sep)

    @classmethod
    def from_paths(cls, paths, sep="/"):
        trie = cls(sep)
        for path in paths:
            trie.insert(path)
        trie.compact()
        return trie
