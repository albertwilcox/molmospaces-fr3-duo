"""Driver class for SpaceMouse controller. Modified based on the robosuite code.

This class provides a driver support to SpaceMouse on Mac OS X.
In particular, we assume you are using a SpaceMouse Wireless by default.

To set up a new SpaceMouse controller:
    1. Download and install driver from https://www.3dconnexion.com/service/drivers.html
    2. Install hidapi library through pip
       (make sure you run uninstall hid first if it is installed).
    3. Make sure SpaceMouse is connected before running the script
    4. (Optional) Based on the model of SpaceMouse, you might need to change the
       vendor id and product id that correspond to the device.

For Linux support, you can find open-source Linux drivers and SDKs online.
    See http://spacenav.sourceforge.net/

"""

import os
import sys
import time
from collections import namedtuple

import numpy as np

try:
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. "
        "Only Mac OS X is officially supported. Install the additional "
        "requirements with `pip install -r requirements-ik.txt`"
    ) from exc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scipy.spatial.transform import Rotation as R

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "y": AxisSpec(channel=1, byte1=1, byte2=2, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


def nms_max_axis(control: np.ndarray, threshold=0.6):
    """
    Suppress all but the axis with the maximum |value|.
    The max axis is set to -1 or 1 based on sign, others are zeroed.

    Args:
        control (np.ndarray): 6D input vector, assumed scaled in [-1, 1]
        threshold (float): minimum |value| to count as valid input

    Returns:
        np.ndarray: filtered control vector with only max direction
    """
    if np.all(np.abs(control) < threshold):
        return np.zeros_like(control)

    max_idx = np.argmax(np.abs(control))
    out = np.zeros_like(control)
    out[max_idx] = np.sign(control[max_idx])
    return np.array(out)


class Keyboard:
    """
    A minimalistic driver class for Keyboard with HID library.
    """

    def __init__(self) -> None:
        pass

    def callback(self, event) -> None:
        pass


class SpaceMouse:
    """
    A minimalistic driver class for SpaceMouse with HID library.

    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.

    Args:
        vendor_id (int): HID device vendor id
        product_id (int): HID device product id
        pos_step (float): step for each position update
        rot_step (float): step for each rotation update
    """

    def __init__(self, vendor_id=0x256F, product_id=0xC652, pos_step=0.05, angle_step=0.05) -> None:
        print("Opening SpaceMouse device")
        self.device = hid.device()
        self.device.open(vendor_id, product_id)  # SpaceMouse
        self.device.set_nonblocking(True)
        self.pos_step = pos_step
        self.angle_step = angle_step

        print(f"Manufacturer: {self.device.get_manufacturer_string()}")
        print(f"Product: {self.device.get_product_string()}")

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.gripper = 0.0

        self._control = np.zeros(6)
        self.mode = "position"

    def update(self, goal_tcp_pos, goal_tcp_quat):
        d = self.device.read(7)

        if len(d) > 0:
            if d[0] == 1:  ## readings from 6-DoF sensor
                if self.mode == "position":
                    self.x = (
                        convert(d[SPACE_MOUSE_SPEC["x"].byte1], d[SPACE_MOUSE_SPEC["x"].byte2])
                        * SPACE_MOUSE_SPEC["x"].scale
                    )
                    self.y = (
                        convert(d[SPACE_MOUSE_SPEC["y"].byte1], d[SPACE_MOUSE_SPEC["y"].byte2])
                        * SPACE_MOUSE_SPEC["y"].scale
                    )
                    self.z = (
                        convert(d[SPACE_MOUSE_SPEC["z"].byte1], d[SPACE_MOUSE_SPEC["z"].byte2])
                        * SPACE_MOUSE_SPEC["z"].scale
                    )
                    self.roll, self.pitch, self.yaw = 0, 0, 0
                else:
                    self.x, self.y, self.z = 0, 0, 0
                    self.roll = (
                        convert(
                            d[SPACE_MOUSE_SPEC["roll"].byte1], d[SPACE_MOUSE_SPEC["roll"].byte2]
                        )
                        * SPACE_MOUSE_SPEC["roll"].scale
                    )
                    self.pitch = (
                        convert(
                            d[SPACE_MOUSE_SPEC["pitch"].byte1], d[SPACE_MOUSE_SPEC["pitch"].byte2]
                        )
                        * SPACE_MOUSE_SPEC["pitch"].scale
                    )
                    self.yaw = (
                        convert(d[SPACE_MOUSE_SPEC["yaw"].byte1], d[SPACE_MOUSE_SPEC["yaw"].byte2])
                        * SPACE_MOUSE_SPEC["yaw"].scale
                    )

                self._control = np.array(
                    nms_max_axis(
                        np.array([self.x, self.y, self.z, self.yaw, self.roll, self.pitch])
                    )
                )
                # print(self._control)

            elif d[0] == 3:  ## readings from the side buttons
                # press left button
                if d[1] == 1:
                    self.gripper = 0.0 if self.gripper == 255.0 else 255.0

                # right button is for reset
                if d[1] == 2:
                    if self.mode == "position":
                        self.mode = "rotation"
                    else:
                        self.mode = "position"

        else:
            return goal_tcp_pos, goal_tcp_quat, self.gripper

        if not isinstance(self._control, np.ndarray):
            self._control = np.array(self._control)

        goal_tcp_pos += self._control[:3] * self.pos_step

        if np.any(self._control[3:]):
            R_increment = R.from_euler("zxy", self._control[3:] * self.angle_step, degrees=False)
            current_tcp_quat = R.from_quat(np.roll(goal_tcp_quat, -1))
            new_tcp_quat = R.from_matrix(
                current_tcp_quat.as_matrix() @ R_increment.as_matrix()
            ).as_quat()
            goal_tcp_quat = np.roll(new_tcp_quat, 1)

        return goal_tcp_pos, goal_tcp_quat, self.gripper

    def reset(self) -> None:
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.gripper = 0.0

        self._control = np.zeros(6)
        self.mode = "position"

    def close(self) -> None:
        """Close the SpaceMouse device."""
        self.device.close()


if __name__ == "__main__":
    space_mouse = SpaceMouse(product_id=50741)
    while True:
        space_mouse.update()
        time.sleep(0.1)
