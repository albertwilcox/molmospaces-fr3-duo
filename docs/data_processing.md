# Data postprocessing

After generating data, it must be postprocessed. This file describes the postprocessing pipeline, in the order in which it must be run.

For an example of the full pipeline, see [this script](/scripts/data/process_data.sh).

## Video path repair

Due to a [known issue](https://github.com/allenai/molmo-spaces/issues/286), `traj_{i}/obs/sensor_data/` in each data file is not being populated with the video filenames, but downstream scripts expect this data in order to consume video information. Until this bug is fixed, use `scripts/data/repair_video_paths.py` to repair the data files.

## Episode validation

For a variety of reasons, generated data could be invalid or corrupted. Running `scripts/data/validate_trajectories.py` finds these episodes and marks them as invalid so future processing or training can filter them out. It does so by adding a `valid_traj_mask` value to the `.h5` file, which is a `(N,)` bool array, where the `i`-th element indicates if `traj_{i}` is a valid (usable) trajectory.

Due to a [known issue](https://github.com/allenai/molmo-spaces/issues/293), the generated data does not properly check for object visibility in the exo camera in some cases. This can be caught and filtered out during postprocessing by using the `--check-visibility` flag.

### Valid trajectory index

The trajectory validation script also builds an index of valid trajectories and saves it in the dataset root as `valid_trajectory_index.json`, with the following structure.
If a house or datafile does not have any valid trajectories, it is not included in the index.
Note that `{datafile_path}` is the path of the datafile relative to the dataset root, and that `traj_len` has **not** trimmed any dummy actions.

```
{
    "house_{house_id}": {
        "{datafile_path}": {
            "traj_{traj_idx}": {traj_len},
            ...
        }
        ...
    },
    ...
}
```

## Statistics computation

It is often desirable to use data statistics for normalization during training. This step of the data postprocessing pipeline (using `scripts/data/calculate_stats.py`) calculates these statistics and saves them with the data for future loading. It does so by adding a `stats/` group to the `.h5` file, with the following structure:

```
stats/
    traj_{ep_idx}/
        {dict_key}: list[dict]
        {array_key}/
            min:
            max:
            mean:
            var:
            ...
```

Essentially, it recreates the heirarchical structure of `traj_{i}` under `stats/`, but replaces values with the statistics. For `dict`-valued items, the statistics are stored in `dict`s, or lists thereof. For array-valued items, a corresponding group is created under `stats/`, with the statistics being created as values in that group.

### Aggregated statistics

The stats calculation script also aggregates the calculated statistics and saves it in the dataset root in `aggregated_statistics.json`.
The structure of this file is as follows, where the data for `<key>` is stored in `traj_{traj_idx}/{key}` in the HDF5 datafile.

```json
{
    "<key>/<move_group>": {
        "mean": [],
        "std": [],
        "min": [],
        "max": [],
        "count": 0,
        "sum": [],
        "sum_sq": []
    }
}
```

## Final h5 root structure

The data postprocessing doesn't modify any data existing in the h5 files, it only adds new groups and values to the root of the file. After this pipeline, the root keys of the h5 file are:

```
traj_{i}/
...
stats/
valid_traj_mask
```
