import mujoco
import numpy as np


def state_from_data(
    data: mujoco.MjData,
    sig: int = mujoco.mjtState.mjSTATE_INTEGRATION,
) -> np.ndarray:
    model = data.model
    state_size = mujoco.mj_stateSize(model, sig)
    state_output = np.empty((state_size, 1), dtype=np.float64)
    mujoco.mj_getState(model, data, state_output, sig)
    return state_output


def data_from_state(
    model: mujoco.MjModel,
    state: np.ndarray,
    sig: int = mujoco.mjtState.mjSTATE_INTEGRATION,
) -> mujoco.MjData:
    assert state.dtype == np.float64, f"State must be of type {np.float64}, not {state.dtype}"
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, sig)
    mujoco.mj_forward(model, data)
    return data
