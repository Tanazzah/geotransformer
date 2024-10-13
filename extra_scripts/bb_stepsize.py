import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Function to load the ply files
def load_ply(file_path):
    mesh = o3d.io.read_point_cloud(file_path)
    return mesh

# Function to create a bounding box
def create_bounding_box(mesh, color):
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox.color = color  # Assign the color
    return bbox

# Function to compute the overlapping bounding box
def compute_overlap_bbox(bbox1, bbox2):
    min_bound = np.maximum(bbox1.min_bound, bbox2.min_bound)
    max_bound = np.minimum(bbox1.max_bound, bbox2.max_bound)
    
    # If there is no overlap, the bounding box will collapse
    if np.any(min_bound >= max_bound):
        return None
    
    # Create the overlapping bounding box
    overlap_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return overlap_bbox

# Function to crop the point cloud based on bounding box
def crop_point_cloud(pcd, bbox):
    return pcd.crop(bbox)

# Function to color the point cloud with a given color
def color_point_cloud(pcd, color):
    pcd.paint_uniform_color(color)
    return pcd

# Function to shift a point cloud along the longitudinal axis (assumed x-axis)
def shift_point_cloud(pcd, shift_value, axis=0):
    translation_vector = np.zeros(3)
    translation_vector[axis] = shift_value  # Shift along the specified axis
    pcd.translate(translation_vector)
    return pcd

# Function to calculate point cloud similarity (based on overlap ratio)
def compute_similarity(pcd1, pcd2):
    distance = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    similarity = np.mean(distance)
    return similarity

# Function to visualize point clouds and bounding boxes
def visualize_point_clouds(pcds, bboxes):
    o3d.visualization.draw_geometries(pcds + bboxes)

def main(ply_file_1, ply_file_2):
    # Load the ply files
    pcd1 = load_ply(ply_file_1)
    pcd2 = load_ply(ply_file_2)
    
    # Create bounding boxes with different colors
    bbox1 = create_bounding_box(pcd1, [1, 0, 0])  # Red for the first ply
    bbox2 = create_bounding_box(pcd2, [0, 1, 0])  # Green for the second ply
    
    # Visualize before shifting
    print("Visualizing point clouds before shifting...")
    visualize_point_clouds([pcd1, pcd2], [bbox1, bbox2])

    # Define the range of step sizes for shifting
    step_sizes = np.arange(0.05, 1, 0.005)  # Shifting from 0 to 5 units with step size of 0.5
    correlations = []  # List to store correlation values for each step size

    # Iterate over different step sizes
    for step in step_sizes:
        # Reset the second point cloud
        pcd2_shifted = load_ply(ply_file_2)

        # Shift ply_file_2 along the longitudinal axis (x-axis) by the current step size
        pcd2_shifted = shift_point_cloud(pcd2_shifted, shift_value=step, axis=0)

        # Recompute the bounding box after shifting
        bbox2_shifted = create_bounding_box(pcd2_shifted, [0, 1, 0])  # Green for the shifted ply
        
        # Compute the overlapping bounding box
        overlap_bbox = compute_overlap_bbox(bbox1, bbox2_shifted)
        if overlap_bbox is None:
            correlations.append(np.inf)  # No overlap, assign a high value for dissimilarity
            continue
        
        # Crop the point clouds based on the overlapping region
        cropped_pcd1 = crop_point_cloud(pcd1, overlap_bbox)
        cropped_pcd2 = crop_point_cloud(pcd2_shifted, overlap_bbox)
        
        # Compute the similarity/correlation metric (mean distance between overlapping points)
        similarity = compute_similarity(cropped_pcd1, cropped_pcd2)
        correlations.append(similarity)
    
    # Plot the step size vs correlation graph
    plt.figure(figsize=(10, 6))
    plt.plot(step_sizes, correlations, marker='o', linestyle='-', color='b')
    plt.title('Step Size vs Correlation (Mean Distance)')
    plt.xlabel('Step Size')
    plt.ylabel('Correlation (Lower is Better)')
    plt.grid(True)
    plt.show()

    # Find the best step size (minimum correlation value)
    best_step = step_sizes[np.argmin(correlations)]
    print(f"Best step size: {best_step} with correlation: {min(correlations)}")

    # Visualize after shifting to the best step size
    print("Visualizing point clouds after shifting to the best step size...")
    pcd2_best_shifted = load_ply(ply_file_2)
    pcd2_best_shifted = shift_point_cloud(pcd2_best_shifted, shift_value=best_step, axis=0)
    
    # Recompute the bounding box after the best shift
    bbox2_best_shifted = create_bounding_box(pcd2_best_shifted, [0, 1, 0])  # Green for the shifted ply
    
    # Visualize the result after shifting to the best step
    visualize_point_clouds([pcd1, pcd2_best_shifted], [bbox1, bbox2_best_shifted])

if __name__ == "__main__":
    # Replace with the actual paths of the .ply files
    ply_file_1 = "global_normalize/10.ply"
    ply_file_2 = "global_normalize/11.ply"
    
    main(ply_file_1, ply_file_2)

