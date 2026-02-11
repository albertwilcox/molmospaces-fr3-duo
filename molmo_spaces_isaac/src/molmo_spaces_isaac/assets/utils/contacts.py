import mujoco as mj
from pxr import UsdPhysics

from .data import BaseConversionData, Tokens


def convert_contact_excludes(data: BaseConversionData) -> None:
    """
    Converts MuJoCo contact exclusions (<exclude>) to USD PhysicsFilteredPairsAPI.
    """
    if not data.spec.excludes:
        return

    # We use the Physics prim because collision logic belongs in the Physics layer.
    body_map = data.references.get(Tokens.PHYSICS)
    if body_map is None:
        print("[WARN] No physics body map found in ConversionData. Skipping contact excludes.")
        return

    for exclude in data.spec.excludes:
        if not isinstance(exclude, mj.MjsExclude):
            continue

        b1_name = exclude.bodyname1
        b2_name = exclude.bodyname2

        prim1 = body_map.get(b1_name)
        prim2 = body_map.get(b2_name)

        if prim1 and prim2:
            _apply_filtered_pair(prim1, prim2)
            _apply_filtered_pair(prim2, prim1)
        else:
            # Common if bodies were skipped or are not rigid bodies
            # print(f"not found: {b1_name} - {b2_name}")
            pass


def _apply_filtered_pair(prim_src, prim_target):
    """
    Applies PhysicsFilteredPairsAPI to prim_src and adds prim_target to the list.
    """
    # Ensure the API is applied to the prim (in the Physics layer)
    if not prim_src.HasAPI(UsdPhysics.FilteredPairsAPI):
        api = UsdPhysics.FilteredPairsAPI.Apply(prim_src)
    else:
        api = UsdPhysics.FilteredPairsAPI(prim_src)

    # Add relationship
    api.CreateFilteredPairsRel().AddTarget(prim_target.GetPath())
