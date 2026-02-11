import mujoco
import numpy as np
import open3d as o3d

from molmo_spaces.env.mj_extensions import MjModelBindings


def dump_agent_mesh(
    model_bindings: MjModelBindings,
    model_data: mujoco.MjData,
    output_filename: str = "agent_geometry.obj",
    namespace: str = "robot_0/",
) -> None:
    model = model_bindings.model

    mujoco.mj_step(model, model_data)
    body_ids = {id for id, name in model_bindings.body_id2name.items() if namespace in name}

    vertices = []
    faces = []

    for i in range(model.ngeom):
        if model.geom_bodyid[i] not in body_ids or model.geom_type[i] != 7:
            continue

        mesh_id = model.geom_dataid[i]

        num_vertices = len(vertices)
        start_face = model.mesh_faceadr[mesh_id]
        end_face = start_face + model.mesh_facenum[mesh_id]
        mesh_faces = model.mesh_face[start_face:end_face].reshape(-1, 3) + num_vertices
        faces.extend(mesh_faces)

        start_vert = model.mesh_vertadr[mesh_id]
        end_vert = start_vert + model.mesh_vertnum[mesh_id]
        mesh_vertices = np.array(model.mesh_vert[start_vert:end_vert].reshape(-1, 3))

        r = np.array(model_data.geom_xmat[i])
        t = np.array(model_data.geom_xpos[i])
        mesh_vertices = mesh_vertices @ r.reshape(3, 3).transpose() + t.reshape(-1, 3)

        vertices.extend(mesh_vertices)

    with open(output_filename, "w") as file:
        file.write("# OBJ file\n")
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            file.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")  # OBJ indexing starts at 1


def get_geom_mesh_data(
    model_bindings: MjModelBindings,
    model_data: mujoco.MjData,
    geom_name: str,
):
    model = model_bindings.model

    geom_id = model_bindings.geom_name2id[geom_name]

    mesh_id = model.geom_dataid[geom_id]

    start_face = model.mesh_faceadr[mesh_id]
    end_face = start_face + model.mesh_facenum[mesh_id]
    faces = model.mesh_face[start_face:end_face].reshape(-1, 3)

    start_vert = model.mesh_vertadr[mesh_id]
    end_vert = start_vert + model.mesh_vertnum[mesh_id]
    mesh_vertices = np.array(model.mesh_vert[start_vert:end_vert].reshape(-1, 3))

    r = np.array(model_data.geom_xmat[geom_id])
    t = np.array(model_data.geom_xpos[geom_id])
    vertices = mesh_vertices @ r.reshape(3, 3).transpose() + t.reshape(-1, 3)

    return dict(faces=faces, vertices=vertices)


def oriented_bbox_from_vertices(vertices: np.ndarray) -> o3d.geometry.OrientedBoundingBox:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(vertices)
    return pc.get_oriented_bounding_box()


def long_axis_from_oriented_bbox(bbox: o3d.geometry.OrientedBoundingBox) -> np.ndarray:
    rot = np.asarray(bbox.R)
    ext = np.asarray(bbox.extent)
    longest_idx = np.argmax(ext)
    return rot[:, longest_idx]


def is_long_axis_vertical(axis: np.ndarray) -> bool:
    xylen = np.sqrt((axis[:2] ** 2).sum())
    zlen = abs(axis[2])
    return zlen > xylen


def is_geom_long_axis_vertical(
    model_bindings: MjModelBindings,
    model_data: mujoco.MjData,
    geom_name: str,
) -> bool:
    mesh_data = get_geom_mesh_data(model_bindings, model_data, geom_name)
    obox = oriented_bbox_from_vertices(mesh_data["vertices"])
    axis = long_axis_from_oriented_bbox(obox)
    return is_long_axis_vertical(axis)
