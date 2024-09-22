import open3d as o3d
import numpy as np
import os
def apply_transform(points: np.ndarray, transform: np.ndarray, normals: np.ndarray= None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points

import os

def apply_and_save_transforms(transform_file_path, source_ply_folder):
    # Load transformation matrices from file
    with open(transform_file_path, 'r') as file:
        transform_matrices = []
        transform = []
        for line in file:
            if line.strip():
                transform.append([float(x) for x in line.strip().split()])
            else:
                transform_matrices.append(np.array(transform))
                transform = []

    # Get list of source ply files from the folder
    source_ply_files = os.listdir(source_ply_folder)
    source_ply_paths = [os.path.join(source_ply_folder, file) for file in source_ply_files]

    # Apply each transformation matrix to the corresponding source point cloud
    for i, (transform_matrix, source_ply_path) in enumerate(zip(transform_matrices, source_ply_paths)):
        # Load source point cloud
        source_pcd = o3d.io.read_point_cloud(source_ply_path)

        # Convert point cloud to numpy array
        src_points = np.asarray(source_pcd.points)

        # Apply transformation
        src_points_transformed = apply_transform(src_points, transform_matrix)

        # Save transformed point cloud
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(src_points_transformed)
        o3d.io.write_point_cloud(f"/home/tanazzah/GeoTransformer/data/dentskan/ply2/new{i}.ply", transformed_pcd)

# Example usage
transform_file_path = "transform_matrices.txt"  # Path to the file containing transformation matrices
source_ply_folder = "ply/raw"  # Folder containing source ply files


apply_and_save_transforms(transform_file_path, source_ply_folder)
