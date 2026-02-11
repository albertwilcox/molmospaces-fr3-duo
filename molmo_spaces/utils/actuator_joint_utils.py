from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
from mujoco import MjData, MjModel

from molmo_spaces.env.mj_extensions import MjModelBindings


def extract_joint_names_and_indices(
    model_bindings: MjModelBindings,
    namespace: str = "robot_0/",
    name_filter: Callable[[str], bool] | None = None,
):
    model = model_bindings.model

    if name_filter is None:

        def name_filter(name) -> bool:
            return True

    robot_joint_names = [
        jname
        for jname in model_bindings.joint_name2id
        if jname.startswith(namespace) and name_filter(jname)
    ]

    robot_joint_ids = [model_bindings.joint_name2id[jname] for jname in robot_joint_names]

    robot_joint_types = [model.jnt_type[jid] for jid in robot_joint_ids]
    robot_joint_addr = [model.jnt_qposadr[jid] for jid in robot_joint_ids]
    robot_joint_lengths = [
        (
            7
            if jtype == mujoco.mjtJoint.mjJNT_FREE
            else (4 if jtype == mujoco.mjtJoint.mjJNT_BALL else 1)
        )
        for jtype in robot_joint_types
    ]

    return dict(
        names=robot_joint_names,
        ids=robot_joint_ids,
        addr=robot_joint_addr,
        length=robot_joint_lengths,
    )


def add_joint_qpos(data: MjData, joint_index_dict: dict[str, Any]):
    """`joint_index_dict` can be obtained from `extract_joint_names_and_indices`"""

    joint_index_dict["qpos"] = [
        data.qpos[jaddr : jaddr + jlen].copy()
        for jaddr, jlen in zip(joint_index_dict["addr"], joint_index_dict["length"])
    ]

    return joint_index_dict["qpos"]


def get_tendon_joints(model: MjModel, tendon_id: int):
    """Returns a list of joints affected by a given tendon."""
    joint_ids = []

    # Loop through the wrap objects of this tendon
    for wrap_idx in range(model.tendon_wrapadr[tendon_id], model.tendon_wrapadr[tendon_id + 1]):
        wrap_obj_id = model.tendon_wrapid[wrap_idx]  # ID of the wrap object
        wrap_type = model.wrap_type[wrap_idx]  # Type of the wrap object

        # Check if the wrap object is a joint
        if wrap_type == mujoco.mjtWrap.mjWRAP_JOINT:
            joint_ids.append(wrap_obj_id)

    return joint_ids


def get_tendons_affecting_joint(joint_id, model):
    """Returns a list of tendons that interact with a given joint."""
    affected_tendons = []

    for tendon_id in range(model.ntendon):
        # Get the range of wrap objects for this tendon
        start_idx = model.tendon_adr[tendon_id]
        end_idx = start_idx + model.tendon_num[tendon_id]

        # Loop through the wrap objects
        for wrap_idx in range(start_idx, end_idx):
            if model.wrap_type[wrap_idx] == mujoco.mjtWrap.mjWRAP_JOINT:
                if model.wrap_objid[wrap_idx] == joint_id:
                    affected_tendons.append(tendon_id)
                    break  # No need to check further for this tendon

    return affected_tendons


def actuators_affecting_joint_via_tendon(joint_id: int, model: MjModel):
    # Find tendons affecting this joint
    tendons = get_tendons_affecting_joint(joint_id, model)

    # Find actuators affecting these tendons
    return [act_id for act_id in range(model.nu) if model.actuator_trnid[act_id, 0] in tendons]


def actuators_affecting_joint(joint_id: int, model: MjModel):
    # Find all actuators that affect this joint
    return [act_id for act_id in range(model.nu) if model.actuator_trnid[act_id, 0] == joint_id]


# def get_actuator_type(model, actuator_id):
#     """Returns the type of a given actuator as a string."""
#     actuator_type = model.actuator_dyntype[actuator_id]  # Get actuator type ID
#
#     # MuJoCo defines actuator types as constants in `mujoco.mjtDyn`
#     actuator_types = {
#         mujoco.mjtDyn.mjDYN_NONE: "none",
#         mujoco.mjtDyn.mjDYN_INTEGRATOR: "position",
#         mujoco.mjtDyn.mjDYN_FILTER: "velocity",
#         mujoco.mjtDyn.mjDYN_MUSCLE: "muscle",
#         mujoco.mjtDyn.mjDYN_USER: "user-defined",
#     }
#
#     return actuator_types.get(actuator_type, "unknown")
def get_actuator_type(model, actuator_id, tol=1e-10) -> str:
    """Determine if an actuator is a position or velocity actuator assuming dyntype=None."""

    if model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE:
        print(
            f"Unsupported actuator_dyntype {model.actuator_dyntype[actuator_id]}. Returning unknown actuator type"
        )
        return "unknown"

    gain = abs(model.actuator_gainprm[actuator_id, 0])
    bias_pos = abs(model.actuator_biasprm[actuator_id, 1])
    bias_vel = abs(model.actuator_biasprm[actuator_id, 2])

    if gain > tol:
        if bias_pos > tol:
            return "position"
        elif bias_vel > tol:
            return "velocity"

    return "unknown"


def add_position_actuator_names(
    model_bindings: MjModelBindings, data: MjData, joint_index_dict: dict[str, Any]
):
    model = model_bindings.model

    joint_qpos = add_joint_qpos(data, joint_index_dict)

    actuator_names = []
    for jid, _jqpos in zip(joint_index_dict["ids"], joint_qpos):
        via_tendon = actuators_affecting_joint_via_tendon(jid, model)
        direct = actuators_affecting_joint(jid, model)

        if len(via_tendon + direct) > 1:
            raise NotImplementedError("Only one actuator per joint supported")

        elif len(via_tendon + direct) == 0:
            actuator_names.append(None)
            continue

        act_id = (direct + via_tendon)[0]
        actuator_names.append(model_bindings.actuator_id2name[act_id])

    joint_index_dict["actuator_names"] = actuator_names

    return joint_index_dict["actuator_names"]


def joint_init_dict_merged_by_actuator(data):
    merged = {}
    seen_actuators = {}

    for name, qpos, actuator in zip(data["names"], data["qpos"], data["actuator_names"]):
        if actuator is None:
            merged[name] = {"qpos": qpos.tolist(), "actuator_name": actuator}
        else:
            if actuator in seen_actuators:
                merged[seen_actuators[actuator]]["qpos"] = np.concatenate(
                    (merged[seen_actuators[actuator]]["qpos"], qpos)
                ).tolist()
            else:
                merged[name] = {"qpos": qpos.tolist(), "actuator_name": actuator}
                seen_actuators[actuator] = name

    return merged
