import open3d as o3d
import numpy as np

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

def main(ply_file_1, ply_file_2):
    # Load the ply files
    pcd1 = load_ply(ply_file_1)
    pcd2 = load_ply(ply_file_2)
    
    # Create bounding boxes with different colors
    bbox1 = create_bounding_box(pcd1, [1, 0, 0])  # Red for the first ply
    bbox2 = create_bounding_box(pcd2, [0, 1, 0])  # Green for the second ply
    
    # Compute the overlapping bounding box
    overlap_bbox = compute_overlap_bbox(bbox1, bbox2)
    if overlap_bbox is None:
        print("No overlapping region found.")
        return
    
    # Crop the point clouds based on the overlapping region
    cropped_pcd1 = crop_point_cloud(pcd1, overlap_bbox)
    cropped_pcd2 = crop_point_cloud(pcd2, overlap_bbox)
    
    # Color the overlapping region with a light color (light blue)
    color_point_cloud(pcd1, [1, 0, 0])  # Light blue
    color_point_cloud(pcd2, [0, 1, 0])  # Light blue
    
    color_point_cloud(cropped_pcd1, [1, 0, 0])  # Light blue
    color_point_cloud(cropped_pcd2, [0, 1, 0])  # Light blue

    # Visualize the results
    o3d.visualization.draw_geometries([pcd1, pcd2, bbox1, bbox2],
                                      window_name="Original Point Clouds with Bounding Boxes",
                                      width=800, height=600)

    o3d.visualization.draw_geometries([cropped_pcd1, cropped_pcd2, overlap_bbox],
                                      window_name="Cropped and Colored Overlapping Region",
                                      width=800, height=600)
    
    # Optionally save the cropped point clouds
    o3d.io.write_point_cloud("cropped_ply1.ply", cropped_pcd1)
    o3d.io.write_point_cloud("cropped_ply2.ply", cropped_pcd2)
    print("Cropped and colored point clouds saved.")

if __name__ == "__main__":
    # Replace with the actual paths of the .ply files
    ply_file_1 = "ransac/4.ply"
    ply_file_2 = "ransac/5.ply"
    
    main(ply_file_1, ply_file_2)

