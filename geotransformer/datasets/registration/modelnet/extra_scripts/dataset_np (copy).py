import os.path as osp
from typing import Dict, Optional
import os
import numpy as np
import torch.utils.data
import open3d as o3d
from IPython import embed
from plyfile import PlyData
from geotransformer.utils.common import load_pickle
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
    # fmt: off
    ALL_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
        'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
        'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
        'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'curtain', 'desk', 'door', 'dresser',
        'glass_box', 'guitar', 'keyboard', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'tv_stand', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_INDICES = [
        0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36,
        38, 39
    ]
    # fmt: on

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
        self.class_indices = self.get_class_indices(class_indices, asymmetric)
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

        self.data_list = load_pickle(osp.join(dataset_root, f'{subset}.pkl'))
        '''data_list = [x for x in data_list if x['label'] in self.class_indices]
        if overfitting_index is not None and deterministic:
            data_list = [data_list[overfitting_index]]
        self.data_list = data_list'''

    def get_class_indices(self, class_indices, asymmetric):
        r"""Generate class indices.
        'all' -> all 40 classes.
        'seen' -> first 20 classes.
        'unseen' -> last 20 classes.
        list|tuple -> unchanged.
        asymmetric -> remove symmetric classes.
        """
        if isinstance(class_indices, str):
            assert class_indices in ['all', 'seen', 'unseen']
            if class_indices == 'all':
                class_indices = list(range(40))
            elif class_indices == 'seen':
                class_indices = list(range(20))
            else:
                class_indices = list(range(20, 40))
        if asymmetric:
            class_indices = [x for x in class_indices if x in self.ASYMMETRIC_INDICES]
        class_indices=0
        return class_indices

   
    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index

        #data_dict: Dict = self.data_list[index]
        data_dict = self.data_list[index]
        ref_points = data_dict['points0'].copy()
        ref_normals = data_dict['normals0'].copy()
        src_points = data_dict['points1'].copy()
        src_normals = data_dict['normals1'].copy()
        label = data_dict['label']
        raw_points=ref_points.copy()
        raw_normals=ref_normals.copy()
        #refpath=data_dict['pcd0']
        #srcpath=data_dict['pcd1']
        #ref_points = o3d.io.read_point_cloud(os.path.join(rootsave1, refpath))
        #src_points = o3d.io.read_point_cloud(os.path.join(rootsave1, srcpath))

        if self.deterministic:
            np.random.seed(index)
        # normalize raw point cloud
        ref_points = normalize_points(ref_points)
        src_points = normalize_points(src_points)
        
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
            # crop
            '''if self.keep_ratio is not None:
                if self.crop_method == 'plane':
                    ref_points, ref_normals = random_crop_point_cloud_with_plane(
                        ref_points, keep_ratio=self.keep_ratio, normals=ref_normals
                    )
                    src_points, src_normals = random_crop_point_cloud_with_plane(
                        src_points, keep_ratio=self.keep_ratio, normals=src_normals
                    )
                else:
                    viewpoint = random_sample_viewpoint()
                    ref_points, ref_normals = random_crop_point_cloud_with_point(
                        ref_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=ref_normals
                    )
                    src_points, src_normals = random_crop_point_cloud_with_point(
                        src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
                    )'''

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
        
        print(new_data_dict)
        print("INDEXXX", index)
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
	#del my_dict[key]
        
        return new_data_dict

    def __len__(self):
        return len(self.data_list)
