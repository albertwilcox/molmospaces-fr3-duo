import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def usd_quat_to_numpy(quat: Gf.Quatf | Gf.Quatd) -> np.ndarray:
    w = quat.GetReal()
    xyz = quat.GetImaginary()
    return np.array([w, *xyz], dtype=np.float64)


def is_prim_articulated(prim: Usd.Prim) -> tuple[bool, Usd.Prim | None]:
    if not prim.IsValid():
        return False, None

    articulation_root_prim: Usd.Prim | None = None
    is_articulated = False
    for curr_prim in Usd.PrimRange(prim):
        if curr_prim.HasAPI(UsdPhysics.ArticulationRootAPI):  # type: ignore
            is_articulated = True
            articulation_root_prim = curr_prim
            break

    return is_articulated, articulation_root_prim


def set_prim_pose(prim: Usd.Prim, pos: np.ndarray, quat: np.ndarray) -> None:
    if not prim.IsValid():
        return

    xform = UsdGeom.Xformable(prim)
    tf = Gf.Transform(translation=pos.tolist(), rotation=Gf.Rotation(Gf.Quatf(*quat)))
    tf_op = xform.MakeMatrixXform()
    tf_op.Set(tf.GetMatrix())


def get_prim_local_pose(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros(3, np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if not prim.IsValid():
        return pos, quat

    xform = UsdGeom.Xformable(prim)
    local_tf = xform.GetLocalTransformation()
    pos = np.array(local_tf.ExtractTranslation(), dtype=np.float64)
    quat = usd_quat_to_numpy(local_tf.ExtractRotation().GetQuat())

    return pos, quat


def get_prim_world_pose(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros(3, np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if not prim.IsValid():
        return pos, quat

    xform = UsdGeom.Xformable(prim)
    world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = np.array(world_tf.ExtractTranslation(), dtype=np.float64)
    quat = usd_quat_to_numpy(world_tf.ExtractRotation().GetQuat())

    return pos, quat


def compute_bbox_size(prim: Usd.Prim) -> np.ndarray | None:
    if not prim.IsValid():
        print(f"Got an issue when getting bbox from prim '{prim.GetName()}'")
        return None

    boundable = UsdGeom.Boundable(prim)
    bound = boundable.ComputeWorldBound(0, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return np.array(bound_range.GetSize(), dtype=np.float64)


def get_root_rigid_body_prim(prim: Usd.Prim) -> Usd.Prim | None:
    for curr_prim in Usd.PrimRange(prim):
        if curr_prim.HasAPI(UsdPhysics.RigidBodyAPI):  # type: ignore
            return curr_prim
    return None


def get_root_articulation_prim(prim: Usd.Prim) -> Usd.Prim | None:
    for curr_prim in Usd.PrimRange(prim):
        if curr_prim.HasAPI(UsdPhysics.ArticulationRootAPI):  # type: ignore
            return curr_prim

    return None
