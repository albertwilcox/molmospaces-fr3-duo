import os

import prior


# OBJAVERSE_HOUSES_DIR = os.path.abspath("./objaverse_houses/houses_2023_07_28")
def load_objaverse_houses(house_dataset_path, subset_to_load="val"):
    # max_houses_per_split = {"train": int(1e9), "val": int(1e9), "test": int(1e9)}
    max_houses_per_split = {"train": 0, "val": 0, "test": 0}
    print(house_dataset_path)

    max_houses_per_split[subset_to_load] = int(1e9)
    return prior.load_dataset(
        "procthor-objaverse-internal",
        revision="local",
        path_to_splits=None,
        split_to_path={
            k: os.path.abspath(os.path.join(house_dataset_path, f"{k}.jsonl.gz"))
            for k in ["train", "val", "test"]
        },
        max_houses_per_split=max_houses_per_split,
    )[subset_to_load]
