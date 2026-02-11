# Data Format

Data generation produces a folder structure as follows:

```
train/
    house_{house_idx}/
        episode_{ep_idx:08d}_{camera_name}_batch_{batch_idx}_of_{n_batches}.mp4
        ...
        trajectories_batch_{batch_idx}_of_{n_batches}.h5
        ...
    ...
test/
    ...
```

Within each h5 file, the structure is (largely) described as follows. When a type is annotated as `dict`, it means that it is a `(K,)` uint8 array containing a byte-encoded json-encoded dictionary right-padded with `\x00` to `K` bytes. When a type is annotated as `list[dict]` it is a 2D array where each element is a `(K,)` uint8 array. Note that `K` might differ between values within a trajectory, and `T` will differ between trajectories.

Additionally, note that the `i`-th state corresponds to the `i+1`-th action. This means that the first action is dummy and should not be used. Furthermore, a special `done` sentinel action is used during datagen, which is also included as the last action. If one intends only to supervise on move actions (without considering `done` actions), the first and last action should be discarded, and the last 2 states should be discarded. In all cases, however, the first action and last state should be discarded.

```
traj_{ep_idx}/
    actions/
        commanded_action: list[dict]  # the action returned by the policy (e.g. demonstrator), may or may not be positions
        joint_pos: list[dict]  # joint position commands for position-controlled move-groups
        joint_pos_rel: list[dict]  # delta between commanded joint pos and current joint pos for position-controlled move-groups
        ee_pose: list[dict]  # commanded leaf frame rel. to robot base (pos+quat) for position-controlled move-groups
        ee_twist: list[dict]  # commanded 6D body-frame ee twist of leaf frame for position-controlled move groups
    obs/
        agent/
            qpos: list[dict]
            qvel: list[dict]
        extra/ (other state information)
            ...
        sensor_data/
            {camera_name}: byte-encoded string representing the video filename of this camera's feed, in the same directory as the h5 file.
        sensor_param/
            {camera_name}/
                cam2world_gl: (T, 4, 4) array of camera-to-world transformation matrices
                extrinsic_cv: (T, 3, 4) top 3 rows of cam2world_gl
                intrinsic_cv: (T, 3, 3) camera intrinsics matrics
    obs_scene: dict, time-invariant scene-specific information
    rewards: (T,) array of floats, per-step reward
    success: (T,) array of bool, per-step done signal
    terminated: (T,) array of bool, per-step done or timeout signal
    truncated: (T,) array of bool, per-step timeout signal
...
```

## Data postprocessing

After generating data, it must be postprocessed for consumption by a training pipeline. See [the documentation](data_processing.md) for more information.


### Final h5 root structure

The data postprocessing doesn't modify any existing data in the h5 files, it only adds new groups and values to the root of the file.
After this pipeline, the root keys of the h5 file are:

```
traj_{i}/
...
stats/
valid_traj_mask
```
