import os.path as osp
from typing import Dict, Optional
import os
import numpy as np
import torch.utils.data
import open3d as o3d
from IPython import embed
from plyfile import PlyData
from geotransformer.utils.pointcloud import random_sample_transform, apply_transform, inverse_transform, regularize_normals
from geotransformer.utils.registration import compute_overlap
from geotransformer.utils.open3d import estimate_normals, voxel_downsample
from geotransformer.transforms.functional import (
    normalize_points,
    random_jitter_points,
    random_shuffle_points,
    random_sample_points,
    random_crop_point_cloud_with_plane,
    random_sample_viewpoint,
    random_crop_point_cloud_with_point,
)


class ModelNetPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        subset: str,
        num_points: int = 1024,
        voxel_size: Optional[float] = None,
        rotation_magnitude: float = 30.0,
        translation_magnitude: float = 0.7,
        noise_magnitude: Optional[float] = None,
        keep_ratio: float = 0.7,
        crop_method: str = 'plane',
        asymmetric: bool = True,
        class_indices: str = 'all',
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
        overfitting_index: Optional[int] = None,
    ):
        super(ModelNetPairDataset, self).__init__()

        assert subset in ['train', 'val', 'test']
        assert crop_method in ['plane', 'point']

        self.dataset_root = dataset_root
        self.subset = subset

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.asymmetric = asymmetric
        self.class_indices = 0
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal
        self.overfitting_index = overfitting_index

        self.data_files = os.listdir('ply2' )

    def compute_normals(self,points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)
        return normals
    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index

        file_name = self.data_files[index]
        label = 0
        current_directory = os.getcwd()
        print("Current directory:", current_directory)

        rootsave1 = 'ply2'  
        refpath = file_name
        srcpath = file_name  # Assuming the same file for src and ref
        print("src_path",os.path.join(rootsave1, refpath))
        
        ref_points = np.asarray(o3d.io.read_point_cloud(os.path.join(rootsave1, refpath)).points)
        src_points = np.asarray(o3d.io.read_point_cloud(os.path.join(rootsave1, srcpath)).points)
        print(ref_points)
        print(ref_points.shape)
        if self.deterministic:
            np.random.seed(index)
        
        ref_normals=self.compute_normals(ref_points)
        src_normals=self.compute_normals(src_points)
        raw_normals=self.compute_normals(ref_points)
        # normalize raw point cloud
        ref_points = normalize_points(ref_points)
        src_points = normalize_points(src_points)
        raw_points=ref_points.copy()
        # once sample on raw point cloud
        if not self.twice_sample:
            ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
            src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)

        # random transform to source point cloud
        transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
        inv_transform = inverse_transform(transform)
        src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

        raw_ref_points = ref_points
        raw_ref_normals = ref_normals
        raw_src_points = src_points
        raw_src_normals = src_normals

        while True:
            ref_points = raw_ref_points
            ref_normals = raw_ref_normals
            src_points = raw_src_points
            src_normals = raw_src_normals

            # data check
            is_available = True
            # check overlap
            if self.check_overlap:
                overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.05)
                if self.min_overlap is not None:
                    is_available = is_available and overlap >= self.min_overlap
                if self.max_overlap is not None:
                    is_available = is_available and overlap <= self.max_overlap
            if is_available:
                break

        if self.twice_sample:
            # twice sample on both point clouds
            ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
            src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)

        # random shuffle
        ref_points, ref_normals = random_shuffle_points(ref_points, normals=ref_normals)
        src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)

        if self.voxel_size is not None:
            # voxel downsample reference point cloud
            ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
            src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)

        new_data_dict = {
            'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            'label': int(label),
            'index': int(index),
        }
        #print(ref_points)
        rootsave='/home/tanazzah/GeoTransformer/experiments/dentskan/ply'
        ref_ply_path = osp.join(rootsave, f'reference_{index}.ply')
        src_ply_path = osp.join(rootsave, f'source_{index}.ply')
        raw_ply_path = osp.join(rootsave, f'raw_{index}.ply')
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # Suppress Open3D warnings

        ref_cloud = o3d.geometry.PointCloud()
        ref_cloud.points = o3d.utility.Vector3dVector(new_data_dict['ref_points'])
        if 'ref_normals' in new_data_dict:
            ref_cloud.normals = o3d.utility.Vector3dVector(new_data_dict['ref_normals'])
        o3d.io.write_point_cloud(ref_ply_path, ref_cloud)

        src_cloud = o3d.geometry.PointCloud()
        src_cloud.points = o3d.utility.Vector3dVector(new_data_dict['src_points'])
        if 'src_normals' in new_data_dict:
            src_cloud.normals = o3d.utility.Vector3dVector(new_data_dict['src_normals'])
        o3d.io.write_point_cloud(src_ply_path, src_cloud)

        raw_cloud = o3d.geometry.PointCloud()
        raw_cloud.points = o3d.utility.Vector3dVector(new_data_dict['raw_points'])
        if 'raw_normals' in new_data_dict:
            raw_cloud.normals = o3d.utility.Vector3dVector(new_data_dict['raw_normals'])
        o3d.io.write_point_cloud(raw_ply_path, raw_cloud)

        if self.estimate_normal:
            ref_normals = estimate_normals(ref_points)
            ref_normals = regularize_normals(ref_points, ref_normals)
            src_normals = estimate_normals(src_points)
            src_normals = regularize_normals(src_points, src_normals)

        if self.return_normals:
            new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
            new_data_dict['ref_normals'] = ref_normals.astype(np.float32)
            new_data_dict['src_normals'] = src_normals.astype(np.float32)

        if self.return_occupancy:
            new_data_dict['ref_feats'] = np.ones_like(ref_points[:, :1]).astype(np.float32)
            new_data_dict['src_feats'] = np.ones_like(src_points[:, :1]).astype(np.float32)        
        return new_data_dict

    def __len__(self):
        return len(self.data_files)

