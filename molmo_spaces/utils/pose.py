import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_mat_to_7d(pose_matrix: np.ndarray) -> np.ndarray:
    """Convert 4x4 pose matrix to 7D vector (x, y, z ,qw, qx, qy, qz)."""
    assert pose_matrix.shape == (4, 4)
    pos = pose_matrix[:3, 3]
    rot_quat = R.from_matrix(pose_matrix[:3, :3]).as_quat(scalar_first=True)  # Returns [w, x, y, z]
    return np.concatenate([pos, rot_quat])


def pos_quat_to_pose_mat(
    pos: np.ndarray | list, quat: np.ndarray | list | None = None
) -> np.ndarray:
    if quat is None:
        assert len(pos) == 7
        quat = pos[3:7]
        pos = pos[0:3]

    assert len(pos) == 3
    assert len(quat) == 4
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    pose_matrix[:3, 3] = pos
    return pose_matrix


def pose_mat_to_pos_quat(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = pose[:3, 3]
    quat = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
    return pos, quat
