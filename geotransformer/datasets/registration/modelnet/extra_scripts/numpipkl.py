import numpy as np
import os
import open3d as o3d
import pickle

# Function to compute normals for a point cloud
def compute_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    return normals

# Main function to convert ply files to pkl
def convert_ply_to_pkl(index):
    data_list = []
    folder_path="ply2"
    ply_file_path = os.path.join(folder_path, file_list[i])
    points0 = np.asarray(o3d.io.read_point_cloud(ply_file_path).points)
    normals0 = compute_normals(points0)
    points1 = np.asarray(o3d.io.read_point_cloud(os.path.join(folder_path, file_list[i + 1])).points)
    normals1 = compute_normals(points1)
        
    label = 0  # Extracting label from filename
        
    data_dict = {
        'points0': points0,
        'normals0': normals0,
        'points1': points1,
        'normals1': normals1,
        'label': int(label)
    }
        return data_dict
# Example usage:
folder_path = "s3d"
output_file = "test.pkl"
convert_ply_to_pkl(folder_path, output_file)

