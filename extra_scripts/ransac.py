import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    # Downsample the point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.001
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

# Load the point clouds
pcd1 = o3d.io.read_point_cloud("ransac/5.ply")
pcd2 = o3d.io.read_point_cloud("ransac/6.ply")

# Preprocess the point clouds (downsample + estimate normals + compute FPFH)
voxel_size = 0.005  # Set the voxel size based on the scale of the point cloud
pcd1_down, fpfh1 = preprocess_point_cloud(pcd1, voxel_size)
pcd2_down, fpfh2 = preprocess_point_cloud(pcd2, voxel_size)

# Perform RANSAC-based alignment
ransac_result = execute_global_registration(pcd1_down, pcd2_down, fpfh1, fpfh2, voxel_size)

# Apply the resulting transformation matrix to align the second point cloud
transformation_matrix = ransac_result.transformation
pcd2.transform(transformation_matrix)

# Compute the bounding boxes for both point clouds
bbox1 = pcd1.get_axis_aligned_bounding_box()
bbox2 = pcd2.get_axis_aligned_bounding_box()

bbox1.color = (1, 0, 0)  # Red color for bbox1
bbox2.color = (0, 1, 0)  # Green color for bbox2
# Visualize the aligned point clouds and their bounding boxes
pcd1.paint_uniform_color([1, 0, 0])  # Color pcd1 red
pcd2.paint_uniform_color([0, 1, 0])  # Color pcd2 green

o3d.visualization.draw_geometries([pcd1, pcd2, bbox1, bbox2])

