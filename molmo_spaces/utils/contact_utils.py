import mujoco
import numpy as np

from molmo_spaces.env.mj_extensions import MjModelBindings


def contact_force_frame_pos_dist(model: MjModelBindings, data: mujoco.MjData, contact_id: int):
    force_torque = np.array((6,), dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_id, force_torque)
    force_torque = force_torque.flatten().astype(np.float32)

    contact = data.contact[contact_id]

    frame = contact.frame.flatten().astype(np.float32)
    pos = contact.pos.flatten().astype(np.float32)
    dist = contact.dist.flatten().astype(np.float32)

    # 6 + 9 + 3 + 1 = 19-D
    return np.concatenate([force_torque, frame, pos, dist])
