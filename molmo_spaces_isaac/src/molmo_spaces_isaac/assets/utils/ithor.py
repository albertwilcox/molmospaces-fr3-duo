import mujoco as mj
import usdex.core
from pxr import Gf, Usd, UsdPhysics

from molmo_spaces_isaac.assets.utils.data import (
    BaseConversionData,
    BodyData,
    SceneObjectInfo,
    Tokens,
    to_usd_quat,
)
from molmo_spaces_isaac.assets.utils.geom import (
    STRUCTURAL_CLASS,
    VISUAL_CLASS,
    convert_geom,
    is_visual,
)
from molmo_spaces_isaac.assets.utils.joint import convert_joint_flatten
from molmo_spaces_isaac.assets.utils.material import convert_material
from molmo_spaces_isaac.assets.utils.mesh import convert_mesh

# PREFIXES_ARTICULATED_TO_KEEP = ("cabinet_", "drawer_", "dishwasher_")

ITHOR_IBL_COLOR = (1.0, 0.85301, 0.55212)
# ITHOR_IBL_COLOR = (1.0, 1.0, 1.0)

ITHOR_LEAF_GEOMS_DENSITIES: dict[str, float] = {
    "drawer": 10000.0,
    "cabinet": 10000.0,
    "dishwasher": 10000.0,
    "oven": 10000.0,
}


def convert_ithor_meshes(data: BaseConversionData, objects: list[SceneObjectInfo]) -> None:
    def collect_geoms(body: mj.MjsBody, geoms_list: list[mj.MjsGeom]) -> None:
        geoms_list.extend(geom for geom in body.geoms if geom.type == mj.mjtGeom.mjGEOM_MESH)
        for child_body in body.bodies:
            collect_geoms(child_body, geoms_list)

    meshes: dict[str, mj.MjsMesh] = {}
    for scene_obj in objects:
        # if scene_obj.articulated and not any(
        #     substr in scene_obj.spec.name for substr in PREFIXES_ARTICULATED_TO_KEEP
        # ):
        #     continue
        geoms: list[mj.MjsGeom] = []
        collect_geoms(scene_obj.spec, geoms)
        for geom in geoms:
            if mesh_handle := data.spec.mesh(geom.meshname):
                meshes[mesh_handle.name] = mesh_handle

    geometry_scope = data.libraries[Tokens.GEOMETRY].GetDefaultPrim()

    orig_names = list(meshes.keys())
    safe_names = data.name_cache.getPrimNames(geometry_scope, orig_names)

    for orig_name, safe_name in zip(orig_names, safe_names):
        mesh = meshes[orig_name]
        mesh_prim = usdex.core.defineXform(geometry_scope, safe_name).GetPrim()
        data.references[Tokens.GEOMETRY][orig_name] = mesh_prim
        _ = convert_mesh(mesh_prim, mesh, data.spec, normalize_mesh_scale=False)


def convert_ithor_materials(data: BaseConversionData, objects: list[SceneObjectInfo]) -> None:
    def collect_geoms(body: mj.MjsBody, geoms_list: list[mj.MjsGeom]) -> None:
        geoms_list.extend(geom for geom in body.geoms if geom.classname.name == VISUAL_CLASS)
        for child_body in body.bodies:
            collect_geoms(child_body, geoms_list)

    materials: dict[str, mj.MjsMaterial] = {}
    for scene_obj in objects:
        geoms: list[mj.MjsGeom] = []
        collect_geoms(scene_obj.spec, geoms)
        for geom in geoms:
            if mat_handle := data.spec.material(geom.material):
                materials[mat_handle.name] = mat_handle

    materials_scope = data.libraries[Tokens.MATERIALS].GetDefaultPrim()

    orig_names = list(materials.keys())
    safe_names = data.name_cache.getPrimNames(materials_scope, orig_names)

    for orig_name, safe_name in zip(orig_names, safe_names):
        material = materials[orig_name]
        material_prim = convert_material(
            materials_scope, safe_name, material, data.spec, data.libraries[Tokens.MATERIALS]
        ).GetPrim()
        data.references[Tokens.MATERIALS][orig_name] = material_prim


def convert_ithor_objects(  # noqa: PLR0915
    data: BaseConversionData,
    objects: list[SceneObjectInfo],
    collections: dict[str, Usd.CollectionAPI],
    prefix: str = "",
) -> list[Usd.Prim]:
    ithor_objs_prims: list[Usd.Prim] = []
    for obj in objects:
        ithor_obj_prim: Usd.Prim | None = None
        if obj.articulated:
            # if not any(substr in obj.spec.name for substr in PREFIXES_ARTICULATED_TO_KEEP):
            #     continue

            bodies_to_fix: list[tuple[str, str]] = []
            tf = Gf.Transform().SetIdentity()
            col_collection = collections.get(STRUCTURAL_CLASS)
            ithor_obj_prim = convert_body_flatten_articulated(
                data=data,
                body=obj.spec,
                parent_tf=tf,
                is_root=True,
                bodies_to_fix=bodies_to_fix,
                prefix=prefix,
                collection=col_collection,
            )

            joints_per_body: dict[str, list[mj.MjsJoint]] = {}

            def collect_mj_joints(
                body: mj.MjsBody, joints_map: dict[str, list[mj.MjsJoint]]
            ) -> None:
                jnts = [jnt for jnt in body.joints if jnt.type != mj.mjtJoint.mjJNT_FREE]
                if body.name not in joints_map:
                    joints_map[body.name] = []
                joints_map[body.name].extend(jnts)
                for child_body in body.bodies:
                    collect_mj_joints(child_body, joints_map)

            collect_mj_joints(obj.spec, joints_per_body)

            for b_name, joints_of_body in joints_per_body.items():
                if len(joints_of_body) == 0:
                    continue
                elif len(joints_of_body) > 1:
                    print(
                        f"[WARN]: Body '{b_name}' has more than one joint, will use only the first one"
                    )

                asset_id = obj.metadata.asset_id if obj.metadata is not None else ""
                convert_joint_flatten(
                    joints_of_body[0],
                    data,
                    prefix=prefix,
                    asset_id=asset_id,
                    thor_parameters=data.thor_parameters,
                )

            for body_name, parent_name in bodies_to_fix:
                body_data = data.bodies.get(body_name, None)
                parent_data = data.bodies.get(parent_name, None)
                if body_data is None:
                    print(f"[WARN]: target body pair to fix '{body_name}' wasn't found")
                    continue

                body_prim = body_data.body_prim
                body_spec = body_data.body_spec

                if parent_name == "world":
                    continue
                elif parent_data is not None:
                    parent_prim = parent_data.body_prim

                    usd_pos = Gf.Vec3d(body_spec.pos.tolist())
                    usd_quat = Gf.Quatd(*body_spec.quat.tolist())

                    name = data.name_cache.getPrimName(
                        body_prim, UsdPhysics.Tokens.PhysicsFixedJoint
                    )
                    frame = usdex.core.JointFrame(
                        usdex.core.JointFrame.Space.Body0, usd_pos, usd_quat
                    )
                    usdex.core.definePhysicsFixedJoint(
                        body_prim, name, parent_prim, body_prim, frame
                    )
                else:
                    print(
                        f"[WARN]: something went wrong setting fix joint for pair '({body_name},{parent_name})'"
                    )
        else:
            tf = Gf.Transform().SetIdentity()
            col_collection = collections.get(STRUCTURAL_CLASS)
            ithor_obj_prim = convert_body_flatten_non_articulated(
                data, obj.spec, tf, prefix=prefix, collection=col_collection
            )
        if ithor_obj_prim is not None:
            ithor_objs_prims.append(ithor_obj_prim)

    return ithor_objs_prims


def convert_body_flatten_non_articulated(
    data: BaseConversionData,
    body: mj.MjsBody,
    parent_tf: Gf.Transform,
    prefix: str = "",
    collection: Usd.CollectionAPI | None = None,
) -> Usd.Prim:
    geo_scope = (
        data.content[Tokens.GEOMETRY].GetDefaultPrim().GetChild(Tokens.GEOMETRY.value).GetPrim()
    )
    safe_name = data.name_cache.getPrimName(geo_scope, f"{prefix}{body.name}")
    body_prim: Usd.Prim = usdex.core.defineXform(geo_scope, safe_name).GetPrim()

    usd_pos = Gf.Vec3d(body.pos.tolist())
    usd_quat = to_usd_quat(body.quat)
    usd_rot = Gf.Rotation(usd_quat)
    usd_scale = Gf.Vec3d(1.0)

    local_tf = Gf.Transform(translation=usd_pos, rotation=usd_rot, scale=usd_scale)
    world_tf: Gf.Transform = local_tf * parent_tf

    world_pos = world_tf.GetTranslation()
    world_quat = Gf.Quatf(world_tf.GetRotation().GetQuat())

    usdex.core.setLocalTransform(body_prim, translation=world_pos, orientation=world_quat)

    if len(body.geoms) > 0:
        geom_names = [f"{prefix}{geom.name}" for geom in body.geoms]
        safe_names = data.name_cache.getPrimNames(body_prim, geom_names)
        for geom, safe_name in zip(body.geoms, safe_names):
            assert isinstance(geom, mj.MjsGeom)
            convert_geom(body_prim, geom, safe_name, data, None, collection)

    body_over = data.content[Tokens.PHYSICS].OverridePrim(body_prim.GetPath())
    data.references[Tokens.PHYSICS][body.name] = body_over

    data.bodies[body.name] = BodyData(
        body_spec=body,
        body_name=body.name,
        body_safename=safe_name,
        body_tf=world_tf,
        body_prim=body_prim,
    )

    for child_body in body.bodies:
        assert isinstance(child_body, mj.MjsBody)
        convert_body_flatten_non_articulated(
            data=data,
            body=child_body,
            parent_tf=world_tf,
            prefix=prefix,
            collection=collection,
        )

    return body_prim


def convert_body_flatten_articulated(
    data: BaseConversionData,
    body: mj.MjsBody,
    parent_tf: Gf.Transform,
    is_root: bool,
    bodies_to_fix: list[tuple[str, str]],
    prefix: str = "",
    collection: Usd.CollectionAPI | None = None,
) -> Usd.Prim:
    geo_scope = (
        data.content[Tokens.GEOMETRY].GetDefaultPrim().GetChild(Tokens.GEOMETRY.value).GetPrim()
    )
    safe_name = data.name_cache.getPrimName(geo_scope, f"{prefix}{body.name}")
    body_prim: Usd.Prim = usdex.core.defineXform(geo_scope, safe_name).GetPrim()

    usd_pos = Gf.Vec3d(body.pos.tolist())
    usd_quat = to_usd_quat(body.quat)
    usd_rot = Gf.Rotation(usd_quat)
    usd_scale = Gf.Vec3d(1.0)

    local_tf = Gf.Transform(translation=usd_pos, rotation=usd_rot, scale=usd_scale)
    world_tf: Gf.Transform = local_tf * parent_tf

    world_pos = world_tf.GetTranslation()
    world_quat = Gf.Quatf(world_tf.GetRotation().GetQuat())

    usdex.core.setLocalTransform(body_prim, translation=world_pos, orientation=world_quat)

    if len(body.geoms) > 0:
        geom_names = [f"{prefix}{geom.name}" for geom in body.geoms]
        safe_names = data.name_cache.getPrimNames(body_prim, geom_names)
        for geom, safe_name in zip(body.geoms, safe_names):
            assert isinstance(geom, mj.MjsGeom)
            if len(body.bodies) == 0 and not is_visual(geom):
                lemma = geom.name.split("_")[0]
                if lemma in ITHOR_LEAF_GEOMS_DENSITIES:
                    geom.density = ITHOR_LEAF_GEOMS_DENSITIES[lemma]
            convert_geom(
                parent=body_prim,
                geom=geom,
                safe_name=safe_name,
                data=data,
                local_tf=None,
                opt_col_collection=collection,
                normalize_mesh_scale=False,
                override_box_with_mesh=False,
            )

    body_over = data.content[Tokens.PHYSICS].OverridePrim(body_prim.GetPath())
    data.references[Tokens.PHYSICS][body.name] = body_over

    if not is_root:
        _ = UsdPhysics.RigidBodyAPI.Apply(body_over)
        if len(body.geoms) == 0 or all(is_visual(geom) for geom in body.geoms):
            mass_api = UsdPhysics.MassAPI.Apply(body_over)
            mass_api.CreateMassAttr().Set(1e-8)

    if is_root:
        _ = UsdPhysics.ArticulationRootAPI.Apply(body_over)
        if data.use_physx:
            body_over.AddAppliedSchema("PhysxArticulationAPI")

    data.bodies[body.name] = BodyData(
        body_spec=body,
        body_name=body.name,
        body_safename=safe_name,
        body_tf=world_tf,
        body_prim=body_prim,
    )

    if len(body.joints) == 0:
        bodies_to_fix.append((body.name, body.parent.name))

    for child_body in body.bodies:
        assert isinstance(child_body, mj.MjsBody)
        convert_body_flatten_articulated(
            data=data,
            body=child_body,
            parent_tf=world_tf,
            is_root=False,
            bodies_to_fix=bodies_to_fix,
            prefix=prefix,
            collection=collection,
        )

    return body_prim
