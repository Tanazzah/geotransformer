import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the angle between two vectors in degrees
def compute_angle_between_vectors(vec1, vec2):
    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    # Compute dot product
    dot_product = np.dot(vec1_norm, vec2_norm)
    # Clamp dot_product to avoid numerical errors causing domain issues in arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Function to load the ply files with color
def load_color_and_crop_ply(file_path, color, axis='x', crop_start_percent=0, crop_end_percent=1):
    """
    Load a PLY file, apply color, and crop based on specified axis and percentages.

    Parameters:
        file_path (str): Path to the PLY file.
        color (list): RGB color to paint the point cloud.
        axis (str): Axis along which to crop ('x', 'y', or 'z').
        crop_start_percent (float): Percentage (0-1) from the start of the axis to begin cropping.
        crop_end_percent (float): Percentage (0-1) from the start of the axis to end cropping.
    
    Returns:
        o3d.geometry.PointCloud: The cropped point cloud.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.paint_uniform_color(color)  # Apply color to the point cloud
    
    # Get axis-aligned bounding box bounds
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    
    # Select the cropping axis
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    
    # Compute dynamic cropping bounds
    crop_min = min_bound.copy()
    crop_max = max_bound.copy()
    crop_min[axis_idx] = min_bound[axis_idx] + crop_start_percent * (max_bound[axis_idx] - min_bound[axis_idx])
    crop_max[axis_idx] = min_bound[axis_idx] + crop_end_percent * (max_bound[axis_idx] - min_bound[axis_idx])
    
    # Create and apply the bounding box
    cropping_bbox = o3d.geometry.AxisAlignedBoundingBox(crop_min, crop_max)
    cropped_pcd = pcd.crop(cropping_bbox)
    
    return cropped_pcd


def load_and_color_ply(file_path, color):
	pcd = o3d.io.read_point_cloud(file_path)
	pcd.paint_uniform_color(color)  # Apply color to the point cloud
	return pcd

# Function to load the ply files
def load_ply(file_path):
    return o3d.io.read_point_cloud(file_path)

# Function to create a bounding box with color
def create_bounding_box(mesh, color):
    bbox = mesh.get_oriented_bounding_box()
    bbox.color = color
    return bbox

# Function to compute the vector of the longest axis of the bounding box
def get_longest_axis_vector(bbox):
    extents = bbox.extent
    longest_axis_idx = np.argmax(extents)
    # Get direction vector for the longest axis
    direction_vector = bbox.R[:, longest_axis_idx] * extents[longest_axis_idx]
    return direction_vector

# Function to compute the angle bisector of two vectors
def compute_angle_bisector(vec1, vec2):
    # Normalize both vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    # Compute the bisector direction
    bisector = vec1_norm + vec2_norm
    return bisector / np.linalg.norm(bisector)  # Normalize the bisector

# Function to shift a point cloud along a custom direction vector
def shift_point_cloud_custom(pcd, shift_value, direction_vector):
    translation_vector = direction_vector * shift_value
    pcd.translate(translation_vector)
    return pcd

# Function to compute the overlapping bounding box
def compute_overlap_bbox(bbox1, bbox2):
    min_bound = np.maximum(bbox1.get_min_bound(), bbox2.get_min_bound())
    max_bound = np.minimum(bbox1.get_max_bound(), bbox2.get_max_bound())
    
    if np.any(min_bound >= max_bound):
        return None
    
    overlap_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return overlap_bbox

# Function to crop the point cloud based on bounding box
def crop_point_cloud(pcd, bbox):
    return pcd.crop(bbox)

# Function to calculate point cloud similarity
def compute_similarity(pcd1, pcd2, threshold):
    distance = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    thresholded_distances = distance[distance <= threshold]
    if len(thresholded_distances) == 0:
        return float('inf')
    similarity = np.mean(thresholded_distances)
    return similarity

# Function to visualize point clouds, bounding boxes, and vectors
def visualize_point_clouds(pcds, bboxes, vectors=None):
    geometries = pcds + bboxes
    if vectors:
        geometries.extend(vectors)
    o3d.visualization.draw_geometries(geometries)

# Function to compute rotation matrix to align two vectors
def compute_rotation_matrix(vec1, vec2):
    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # Compute the cross product and angle between the vectors
    cross_prod = np.cross(vec1_norm, vec2_norm)
    dot_prod = np.dot(vec1_norm, vec2_norm)
    
    if np.isclose(dot_prod, 1.0):
        # Vectors are already aligned
        return np.eye(3)
    elif np.isclose(dot_prod, -1.0):
        # Vectors are opposite; rotate 180 degrees about an orthogonal axis
        orthogonal_axis = np.array([1, 0, 0]) if not np.isclose(vec1_norm[0], 0) else np.array([0, 1, 0])
        return o3d.geometry.get_rotation_matrix_from_axis_angle(orthogonal_axis * np.pi)
    
    # Compute the skew-symmetric cross-product matrix
    cross_matrix = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])
    
    # Use the Rodrigues' rotation formula
    rotation_matrix = (
        np.eye(3) +
        cross_matrix +
        np.matmul(cross_matrix, cross_matrix) * ((1 - dot_prod) / (np.linalg.norm(cross_prod) ** 2))
    )
    return rotation_matrix

# Update the main function
def main(ply_file_1, ply_file_2,counter):

    axis1='y'
    crop_start1=0
    crop_end1=0.8 # Crop 0-30% of PLY1 along X-axis
    axis2='y'
    crop_start2=0.2 
    crop_end2=1.0 # Crop 70-100% of PLY2 along Y-axis
    if counter>4 and counter<10:
        axis1='x'
        crop_start1=0.35
        crop_end1=1.0 # Crop 0-30% of PLY1 along X-axis
        axis2='x'
        crop_start2=0.0
        crop_end2=0.75 # Crop 70-100% of PLY2 along Y-axis
    # Load the ply files and assign colors
    pcd1 = load_color_and_crop_ply(ply_file_1, [1, 0, 0], axis1, crop_start1, crop_end1)  # Red for pcd1
    pcd2 = load_color_and_crop_ply(ply_file_2, [0, 1, 0], axis2, crop_start2, crop_end2)  # Green for pcd2
    
    
    # Create bounding boxes with different colors
    bbox1 = create_bounding_box(pcd1, [1, 0, 0])  # Red
    bbox2 = create_bounding_box(pcd2, [0, 1, 0])  # Green
    
    # Determine longest axis vectors for each bounding box
    longest_vec1 = get_longest_axis_vector(bbox1)
    longest_vec2 = get_longest_axis_vector(bbox2)
    
    # Compute the rotation matrix to align the longest axis of bbox2 with bbox1
    rotation_matrix = compute_rotation_matrix(longest_vec2, longest_vec1)
    
    # Rotate pcd2 using the computed rotation matrix
    pcd2.rotate(rotation_matrix, center=(0, 0, 0))  # Rotate about the origin
    bbox2_rotated = create_bounding_box(pcd2, [0, 1, 0])  # Update bounding box after rotation
    
    # Compute the angle between the aligned longest axis vectors
    aligned_longest_vec2 = get_longest_axis_vector(bbox2_rotated)
    angle_between_vectors = compute_angle_between_vectors(longest_vec1, aligned_longest_vec2)
    print(f"Angle between aligned longest axes: {angle_between_vectors:.2f} degrees")
    
    # Proceed with angle bisector calculation and visualization
    angle_bisector = compute_angle_bisector(longest_vec1, aligned_longest_vec2)
    
    # Visualize before shifting with longest axes and bisector
    #print("Visualizing point clouds after alignment but before shifting...")
    #vec1_line = o3d.geometry.LineSet(
    #    points=o3d.utility.Vector3dVector([bbox1.center, bbox1.center + longest_vec1]),
    #    lines=o3d.utility.Vector2iVector([[0, 1]])
    #)
    #vec2_line = o3d.geometry.LineSet(
    #    points=o3d.utility.Vector3dVector([bbox2_rotated.center, bbox2_rotated.center + aligned_longest_vec2]),
    #    lines=o3d.utility.Vector2iVector([[0, 1]])
    #)
    #bisector_line = o3d.geometry.LineSet(
    #    points=o3d.utility.Vector3dVector([bbox1.center, bbox1.center + angle_bisector * 0.5]),  # Scale for visibility
    #    lines=o3d.utility.Vector2iVector([[0, 1]])
    #)
    #vec1_line.paint_uniform_color([1, 0, 0])
    #vec2_line.paint_uniform_color([0, 1, 0])
    #bisector_line.paint_uniform_color([0, 0, 1])  # Blue for bisector
    
    #visualize_point_clouds([pcd1, pcd2], [bbox1, bbox2_rotated], [vec1_line, vec2_line, bisector_line])

    # Define step sizes for shifting
    step_sizes = np.arange(0.35, 0.7, 0.005)
    correlations = []

    for step in step_sizes:
        # Reset and shift the second point cloud
        pcd2_shifted = load_color_and_crop_ply(ply_file_2, [0, 1, 0])
        pcd2_shifted.rotate(rotation_matrix, center=(0, 0, 0))  # Apply the same rotation
        pcd2_shifted = shift_point_cloud_custom(pcd2_shifted, step, angle_bisector)
        
        # Recompute bounding box after shift
        bbox2_shifted = create_bounding_box(pcd2_shifted, [0, 1, 0])
        
        # Compute overlap and similarity
        overlap_bbox = compute_overlap_bbox(bbox1, bbox2_shifted)
        if overlap_bbox is None:
            correlations.append(np.inf)
            continue
        
        cropped_pcd1 = crop_point_cloud(pcd1, overlap_bbox)
        cropped_pcd2 = crop_point_cloud(pcd2_shifted, overlap_bbox)
        similarity = compute_similarity(cropped_pcd1, cropped_pcd2, 0.02)
        correlations.append(similarity)
    
    # Plot step size vs correlation
    #plt.figure(figsize=(10, 6))
    #plt.plot(step_sizes, correlations, marker='o', linestyle='-', color='b')
    #plt.title('Step Size vs Correlation (Mean Distance)')
    #plt.xlabel('Step Size')
    #plt.ylabel('Correlation (Lower is Better)')
    #plt.grid(True)
    #plt.show()
    
    # Find and visualize the best step
    best_step = step_sizes[np.argmin(correlations)]
    print(f"Best step size: {best_step} with correlation: {min(correlations)}")

    # Final visualization after optimal shift
    pcd2_best_shifted = load_color_and_crop_ply(ply_file_2, [0, 1, 0],axis2, crop_start2, crop_end2)
    pcd2_best_shifted.rotate(rotation_matrix, center=(0, 0, 0))  # Apply the same rotation
    pcd2_best_shifted = shift_point_cloud_custom(pcd2_best_shifted, best_step, angle_bisector)
    #bbox2_best_shifted = create_bounding_box(pcd2_best_shifted, [0, 1, 0])
    
    #visualize_point_clouds([pcd1, pcd2_best_shifted], [bbox1, bbox2_best_shifted], [bisector_line])
    # Create the premodel folder if it does not exist
    output_folder = "premodel"
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed point clouds
    output_file_1 = os.path.join(output_folder, f"{os.path.basename(ply_file_1)}")
    output_file_2 = os.path.join(output_folder, f"{os.path.basename(ply_file_2)}")
    o3d.io.write_point_cloud(output_file_1, pcd1)
    o3d.io.write_point_cloud(output_file_2, pcd2_best_shifted)

    print(f"Processed files saved: {output_file_1}, {output_file_2}")

import os
if __name__ == "__main__":
    # Path to the directory containing .ply files
    folder_path = "../../data/dentskan/global_normalize"
    output_folder = "../../data/dentskan/premodel"
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed point clouds
    output_file_1 = os.path.join(output_folder, "0.ply")
    # List all .ply files in the directory
    ply_files = [f for f in os.listdir(folder_path) if f.endswith(".ply")]
    print(ply_files)
    # Sort the files if needed, e.g., by filename
    #ply_files.sort()

    # Loop through pairs of files (e.g., pairs of consecutive files)
    for i in range(0, len(ply_files) - 1):
        name1=str(i) + ".ply"
        name2=str(i+1)+".ply"
        # Get the full paths for the pair of files
        ply_file_1 = os.path.join(folder_path, name1)
        ply_file_2 = os.path.join(folder_path, name2)
        
        # Call the main function with each pair of files
        main(ply_file_1, ply_file_2, i+1)
        #plt.close()


