import open3d as o3d
import numpy as np

# Function to load the ply files
def load_ply(file_path):
    mesh = o3d.io.read_point_cloud(file_path)
    return mesh

# Function to create an oriented bounding box
def create_oriented_bounding_box(mesh, color):
    obb = mesh.get_oriented_bounding_box()
    obb.color = color  # Assign the color
    return obb

# Function to find the longest axis of the Oriented Bounding Box
def find_longest_axis(obb):
    # The OBB extents (half-lengths along each axis)
    extents = obb.extent
    longest_axis_idx = np.argmax(extents)  # Find index of the longest axis
    axis_vectors = obb.R[:, longest_axis_idx]  # Get the direction of the longest axis
    return axis_vectors, extents[longest_axis_idx] * 2  # Length is twice the half-length

# Function to compute the rotation matrix to align two vectors
def compute_rotation_matrix(v1, v2):
    # Normalize the input vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Compute the cross product (axis of rotation)
    cross_prod = np.cross(v1, v2)
    sin_angle = np.linalg.norm(cross_prod)
    
    if sin_angle == 0:  # Already aligned
        return np.eye(3)

    # Compute the dot product (cosine of the angle)
    cos_angle = np.dot(v1, v2)
    
    # Compute the skew-symmetric matrix for the cross product
    skew_symmetric = np.array([[0, -cross_prod[2], cross_prod[1]],
                               [cross_prod[2], 0, -cross_prod[0]],
                               [-cross_prod[1], cross_prod[0], 0]])
    
    # Compute the rotation matrix using the Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + skew_symmetric + (np.dot(skew_symmetric, skew_symmetric) * ((1 - cos_angle) / (sin_angle ** 2)))
    
    return rotation_matrix

# Function to rotate and align the second point cloud's longest axis with the first point cloud
def align_point_clouds(pcd1, pcd2, axis1, axis2):
    # Compute the rotation matrix to align axis2 to axis1
    rotation_matrix = compute_rotation_matrix(axis2, axis1)
    
    # Apply the rotation to the second point cloud
    pcd2.rotate(rotation_matrix, center=(0, 0, 0))
    return pcd2, rotation_matrix

# Function to visualize the longest axis
def visualize_longest_axis(obb, axis_direction, axis_length, color=[0, 0, 1]):
    center = obb.center
    line_points = [center, center + axis_direction * axis_length]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    line_set.paint_uniform_color(color)
    return line_set

# Function to create a coordinate frame for origin visualization
def create_coordinate_frame(size=1.0, origin=[0, 0, 0]):
    # Create a coordinate frame to visualize the origin and axes (X=red, Y=green, Z=blue)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

def main(ply_file_1, ply_file_2, output_file):
    # Load the ply files
    pcd1 = load_ply(ply_file_1)
    pcd2 = load_ply(ply_file_2)
    
    # Create oriented bounding boxes with different colors
    obb1 = create_oriented_bounding_box(pcd1, [1, 0, 0])  # Red for the first ply
    obb2 = create_oriented_bounding_box(pcd2, [0, 1, 0])  # Green for the second ply
    
    # Find the longest axis for the first point cloud
    axis_dir1, axis_len1 = find_longest_axis(obb1)
    longest_axis1 = visualize_longest_axis(obb1, axis_dir1, axis_len1, [0, 0, 1])  # Blue for the longest axis
    
    # Find the longest axis for the second point cloud
    axis_dir2, axis_len2 = find_longest_axis(obb2)
    longest_axis2 = visualize_longest_axis(obb2, axis_dir2, axis_len2, [0, 1, 1])  # Cyan for the longest axis
    
    # Create a coordinate frame at the origin
    origin_frame = create_coordinate_frame(size=2)
    
    # Visualize the point clouds, bounding boxes, longest axes, and origin BEFORE transformation
    print("Displaying point clouds before transformation. Close the window to proceed to the next visualization.")
    o3d.visualization.draw_geometries([pcd1, pcd2, obb1, obb2, longest_axis1, longest_axis2, origin_frame],
                                      window_name="Before Transformation",
                                      width=800, height=600)
    
    # Rotate and align the second point cloud with the first
    aligned_pcd2, rotation_matrix = align_point_clouds(pcd1, pcd2, axis_dir1, axis_dir2)
    
    # Create a new Oriented Bounding Box for the rotated second point cloud
    obb2_aligned = create_oriented_bounding_box(aligned_pcd2, [0, 1, 0])
    axis_dir2_aligned, axis_len2_aligned = find_longest_axis(obb2_aligned)
    longest_axis2_aligned = visualize_longest_axis(obb2_aligned, axis_dir2_aligned, axis_len2_aligned, [0, 1, 1])
    
    # Print the rotation matrix and the length of the longest axes
    print(f"Rotation matrix to align the second point cloud:\n{rotation_matrix}")
    print(f"Longest axis length of the first point cloud: {axis_len1}")
    print(f"Longest axis length of the aligned second point cloud: {axis_len2_aligned}")
    
    # Save the aligned second point cloud to a new .ply file
    o3d.io.write_point_cloud(output_file, aligned_pcd2)
    print(f"Aligned second point cloud saved to {output_file}")
    
    # Visualize the point clouds, bounding boxes, longest axes, and origin AFTER transformation
    print("Displaying point clouds after transformation. Close the window when done.")
    o3d.visualization.draw_geometries([pcd1, aligned_pcd2, obb1, obb2_aligned, longest_axis1, longest_axis2_aligned, origin_frame],
                                      window_name="After Transformation",
                                      width=800, height=600)

if __name__ == "__main__":
    # Replace with the actual paths of the .ply files
    ply_file_1 = "8.ply"
    ply_file_2 = "9.ply"
    output_file = "aligned_9.ply"  # Output file path for the aligned second point cloud
    
    main(ply_file_1, ply_file_2, output_file)

