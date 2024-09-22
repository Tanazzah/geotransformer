import numpy as np
import open3d as o3d

def apply_transformation(point_cloud_file, transformation_matrix, output_file):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_file)

    # Convert point cloud to numpy array (Nx3)
    points = np.asarray(pcd.points)

    # Add a column of ones to the points array to make it Nx4 for matrix multiplication
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))

    # Apply the transformation matrix (4x4)
    transformed_points = np.dot(points_homogeneous, transformation_matrix.T)

    # Update the point cloud with transformed points (drop the homogeneous coordinate)
    pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])

    # Save the transformed point cloud
    o3d.io.write_point_cloud(output_file, pcd)

# Example usage
if __name__ == "__main__":
    # Define your 4x4 transformation matrix (rotation + translation)
    transformation_matrix = np.array([
        [1, 0, 0, -0.78],  # Rotation and translation matrix here
        [0, 1, 0, 0.07],
        [0, 0, 1, 0.69],
        [0, 0, 0, 1]
    ])

    input_file = "1.ply"  # Replace with your input point cloud file
    output_file = "1.ply"  # Replace with your desired output file name

    apply_transformation(input_file, transformation_matrix, output_file)

