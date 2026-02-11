from typing import TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


import numpy as np


class PointsDict(TypedDict):
    coord: np.ndarray
    color: NotRequired[np.ndarray]
    class_idx: NotRequired[np.ndarray]
    body_id: NotRequired[np.ndarray]
    ancestor_body_id: NotRequired[np.ndarray]
    point_source_id: NotRequired[np.ndarray]


class ExtraDict(TypedDict):
    all_coord: np.ndarray | None
    all_color: NotRequired[np.ndarray | None]
    all_class_idx: np.ndarray | None
