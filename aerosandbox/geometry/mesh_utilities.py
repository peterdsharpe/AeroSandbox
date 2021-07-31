import aerosandbox.numpy as np
from typing import Tuple


def combine_meshes(
        *meshes: Tuple[np.ndarray, np.ndarray]
):
    if len(meshes) == 1:
        return meshes[0]
    elif len(meshes) == 2:
        points1, faces1 = meshes[0]
        points2, faces2 = meshes[1]

        faces2 += len(points1)

        points = np.concatenate((points1, points2))
        faces = np.concatenate((faces1, faces2))

        return points, faces
    else:
        points, faces = combine_meshes(
            meshes[0],
            meshes[1]
        )
        return combine_meshes(
            (points, faces),
            *meshes[2:]
        )


def convert_mesh_to_polydata_format(
        points,
        faces
):
    faces = [
        [len(face), *face]
        for face in faces
    ]
    faces = np.array(faces)
    faces = np.reshape(faces, -1)
    return points, faces
