import os
import open3d as o3d
import numpy as np

def compute_global_max(input_folder: str) -> float:
    r"""Calculate the global maximum point magnitude across all point clouds in the folder.

    Args:
        input_folder: Path to the folder containing point clouds.

    Returns:
        global_max: The maximum magnitude found across all point clouds.
    """
    global_max = 0
    point_cloud_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]

    for file in point_cloud_files:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(os.path.join(input_folder, file))
        points = np.asarray(pcd.points)
        # Calculate the maximum norm of the points in the current file
        max_norm = np.max(np.linalg.norm(points, axis=1))
        # Update global_max if current max_norm is higher
        if max_norm > global_max:
            global_max = max_norm

    return global_max

def compute_normals_and_save(input_folder: str, output_folder: str, global_max: float):
    r"""Normalize point clouds using the global max and compute normals.

    Args:
        input_folder: Path to the folder containing input point clouds.
        output_folder: Path to save the normalized point clouds with normals.
        global_max: The maximum magnitude to normalize all point clouds.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of point cloud files in input folder
    point_cloud_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]

    for file in point_cloud_files:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(os.path.join(input_folder, file))
        points = np.asarray(pcd.points)
        # Center the point cloud by subtracting the mean
        points -= points.mean(axis=0)
        # Normalize points using the global max
        points /= global_max
        
        # Update the point cloud with normalized points
        pcd.points = o3d.utility.Vector3dVector(points)
        # Compute normals
        pcd.estimate_normals()
        # Save the normalized point cloud with normals
        o3d.io.write_point_cloud(os.path.join(output_folder, file), pcd)

if __name__ == "__main__":
    input_folder = "denture/fragmentcopy"
    output_folder = "denture/fragment"

    # Step 1: Calculate the global maximum from all point clouds
    global_max = compute_global_max(input_folder)

    # Step 2: Normalize point clouds and compute normals
    compute_normals_and_save(input_folder, output_folder, global_max)

