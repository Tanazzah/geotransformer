import os
import open3d as o3d
import numpy as np
from typing import Tuple, List, Optional, Union, Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0  # (0, rot_mag)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform
def compute_normals_and_save(input_folder, output_folder):

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of point cloud files in input folder
    point_cloud_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]

    for file in point_cloud_files:
        # Load point cloud
        pcd1 = o3d.io.read_point_cloud(os.path.join(input_folder, file))
        pcd1 = np.asarray(pcd1.points)
        pcd1 = pcd1 - pcd1.mean(axis=0)
        pcd1 = pcd1/ np.max(np.linalg.norm(pcd1, axis=1))        
        transform = random_sample_transform(40.0, 0.5)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd1)
        # Compute normals
        pcd.estimate_normals()
        pcd1,normal = apply_transform(pcd1, transform, normals=np.asarray(pcd.normals))
        pcd.points = o3d.utility.Vector3dVector(pcd1)
        # Compute normals
        pcd.estimate_normals()
        # Save point cloud with normals
        o3d.io.write_point_cloud(os.path.join(output_folder, file), pcd)

if __name__ == "__main__":
    input_folder = "denture/fragmentcopy"
    output_folder = "denture/fragment"
    compute_normals_and_save(input_folder, output_folder)
