import aerosandbox.numpy as np
from typing import Tuple

"""
Documentation of (points, faces) standard format, which is an unstructured mesh format:

Meshes are given here in the common (points, faces) format. In this format, `points` is a Nx3 array, where each row 
gives the 3D coordinates of a vertex in the mesh. Entries into this array are floating-point, generally speaking.

`faces` is a Mx3 array in the case of a triangular mesh, or a Mx4 array in the case of a quadrilateral mesh. Each row 
in this array represents a face. The entries in each row are integers that correspond to the index of `points` where 
the vertex locations of that face are found. 

"""


def stack_meshes(
        *meshes: Tuple[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes in a series of tuples (points, faces) and merges them into a single tuple (points, faces). All (points,
    faces) tuples are meshes given in standard format.

    Args:
        *meshes: Any number of mesh tuples in standard (points, faces) format.

    Returns: Points and faces of the combined mesh. Standard unstructured mesh format: A tuple of `points` and
    `faces`, where:

        * `points` is a `n x 3` array of points, where `n` is the number of points in the mesh.

        * `faces` is a `m x 3` array of faces if `method` is "tri", or a `m x 4` array of faces if `method` is "quad".

            * Each row of `faces` is a list of indices into `points`, which specifies a face.

    """
    if len(meshes) == 1:
        return meshes[0]
    elif len(meshes) == 2:
        points1, faces1 = meshes[0]
        points2, faces2 = meshes[1]

        faces2 = faces2 + len(points1)

        points = np.concatenate((points1, points2))
        faces = np.concatenate((faces1, faces2))

        return points, faces
    else:
        points, faces = stack_meshes(
            meshes[0],
            meshes[1]
        )
        return stack_meshes(
            (points, faces),
            *meshes[2:]
        )


def convert_mesh_to_polydata_format(
        points: np.ndarray,
        faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PyVista uses a slightly different convention for the standard (points, faces) format as described above. They
    give `faces` as a single 1D vector of roughly length (M*3), or (M*4) in the case of quadrilateral meshing.
    Basically, the mesh displayer goes down the `faces` array, and when it sees a number N, it interprets that as the
    number of vertices in the following face. Then, the next N entries are interpreted as integer references to the
    vertices of the face.

    This has the benefit of allowing for mixed tri/quad meshes.

    Args:
        points: `points` array of the original standard-format mesh
        faces: `faces` array of the original standard-format mesh

    Returns:

        (points, faces), except that `faces` is now in a pyvista.PolyData compatible format.

    """
    faces = [
        [len(face), *face]
        for face in faces
    ]
    faces = np.reshape(np.array(faces), -1)
    return points, faces
