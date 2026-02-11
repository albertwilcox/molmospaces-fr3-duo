"""
Much of this code has been copied from mjinx.configuration._lie.
See: https://github.com/based-robotics/mjinx/blob/main/mjinx/configuration/_lie.py
"""

import jax
import jax.numpy as jnp
import jaxlie
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3, SO3


def get_joint_zero(model: mjx.Model) -> jnp.ndarray:
    """
    Get the zero configuration for all joints in the model.

    :param model: The MuJoCo model.
    :return: An array representing the zero configuration for all joints.
    """
    jnts = []

    with jax.default_device(model.qpos0.device):
        for jnt_id in range(model.njnt):
            jnt_type = model.jnt_type[jnt_id]
            match jnt_type:
                case mj.mjtJoint.mjJNT_FREE:
                    jnts.append(jnp.array([0, 0, 0, 1, 0, 0, 0]))
                case mj.mjtJoint.mjJNT_BALL:
                    jnts.append(jnp.array([1, 0, 0, 0]))
                case mj.mjtJoint.mjJNT_HINGE | mj.mjtJoint.mjJNT_SLIDE:
                    jnts.append(jnp.zeros(1))

        return jnp.concatenate(jnts)


def joint_difference(model: mjx.Model, q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the difference between two joint configurations.

    This function calculates the configuration space difference between two joint states.
    For standard joints (hinge, slide), this is a simple subtraction.
    For quaternion-based joints (free, ball), this uses the proper Lie group operations:

    .. math::

        \\Delta q =
        \\begin{cases}
            q_1 - q_2 & \\text{for standard joints} \\\\
            \\log(q_1 \\cdot q_2^{-1}) & \\text{for quaternion joints}
        \\end{cases}

    where:
        - :math:`q_1, q_2` are the two joint configurations
        - :math:`\\log` is the logarithmic map from SO(3) or SE(3) to their tangent spaces

    :param model: The MuJoCo model.
    :param q1: The first joint configuration.
    :param q2: The second joint configuration.
    :return: The difference between the two configurations.
    """
    jnt_diff = []
    idx = 0
    with jax.default_device(model.qpos0.device):
        for jnt_id in range(model.njnt):
            jnt_type = model.jnt_type[jnt_id]
            match jnt_type:
                case mj.mjtJoint.mjJNT_FREE:
                    q1_pos, q1_quat = q1[idx : idx + 3], q1[idx + 3 : idx + 7]
                    q2_pos, q2_quat = q2[idx : idx + 3], q2[idx + 3 : idx + 7]
                    indices = jnp.array([1, 2, 3, 0])

                    frame1_SE3: SE3 = SE3.from_rotation_and_translation(
                        SO3.from_quaternion_xyzw(q1_quat[indices]),
                        q1_pos,
                    )
                    frame2_SE3: SE3 = SE3.from_rotation_and_translation(
                        SO3.from_quaternion_xyzw(q2_quat[indices]),
                        q2_pos,
                    )

                    jnt_diff.append(jaxlie.manifold.rminus(frame1_SE3, frame2_SE3))
                    idx += 7
                case mj.mjtJoint.mjJNT_BALL:
                    q1_quat = q1[idx : idx + 4]
                    q2_quat = q2[idx : idx + 4]
                    indices = jnp.array([1, 2, 3, 0])

                    frame1_SO3: SO3 = SO3.from_quaternion_xyzw(q1_quat[indices])
                    frame2_SO3: SO3 = SO3.from_quaternion_xyzw(q2_quat[indices])

                    jnt_diff.append(jaxlie.manifold.rminus(frame1_SO3, frame2_SO3))
                    idx += 4
                case mj.mjtJoint.mjJNT_HINGE | mj.mjtJoint.mjJNT_SLIDE:
                    jnt_diff.append(q1[idx : idx + 1] - q2[idx : idx + 1])
                    idx += 1

        return jnp.concatenate(jnt_diff)
