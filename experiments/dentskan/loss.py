import torch
import torch.nn as nn
import numpy as np
from geotransformer.modules.ops import apply_transform, pairwise_distance
from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.registration.metrics import isotropic_transform_error
import matplotlib.pyplot as plt
import open3d as o3d
import time
counter=0
class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rre = cfg.eval.rre_threshold
        self.acceptance_rte = cfg.eval.rte_threshold
        self.transform_matrices = []
        self.transform_input = []
        self.counter=0
        #counter=0
    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]      
        precision=0.2
        return precision
    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        self.transform_input.append(transform.cpu().numpy())
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        #src_corr_points = apply_transform(src_corr_points, transform)        
        precision=0.2
        return precision
        
    @torch.no_grad()    
    def visualizecoarse(self, points1, indices1,points2, indices2, window_name="coarse"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points1)
        pcd.paint_uniform_color([1, 0.706, 0])
        selected_points = pcd.select_by_index(indices1)        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.paint_uniform_color([1, 0, 1])
        selected_points2 = pcd2.select_by_index(indices2)
        o3d.visualization.draw_geometries([selected_points ,selected_points2 ], window_name=window_name)

    @torch.no_grad()    
    def visualizecorr(self, points1,points2, window_name="corresponding"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points1)
        pcd.paint_uniform_color([1, 0.706, 0])               
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.paint_uniform_color([1, 0, 1])               
        o3d.visualization.draw_geometries([pcd,pcd2], window_name=window_name)


    @torch.no_grad()    
    def visualizeknnpoints(self, points1, points2,window_name="coarse"):
        points1 = np.array(points1)
        # Reshape to flatten the array
        points1 = points1.reshape(-1, 3)
        # Create Open3D point cloud
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud1.points = o3d.utility.Vector3dVector(points1)
        point_cloud1.paint_uniform_color([1, 0, 1])
        
        points2 = np.array(points2)
        # Reshape to flatten the array
        points2 = points2.reshape(-1, 3)
        # Create Open3D point cloud
        point_cloud2 = o3d.geometry.PointCloud()
        point_cloud2.points = o3d.utility.Vector3dVector(points2)
        point_cloud2.paint_uniform_color([1, 0.706, 0])
        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud1,point_cloud2], window_name=window_name)

    def visualizematch(self, match):
        matching_scores=np.array(match)
        plt.figure(figsize=(8, 6))
        plt.imshow(matching_scores[:,:,0], cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Matching Score')
        plt.xlabel('Source Nodes')
        plt.ylabel('Reference Nodes')
        plt.title('Matching Scores Heatmap')
        plt.show()
      
    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_p = output_dict['src_points']
        ref_p = output_dict['ref_points']
        src_point = src_p.cpu().numpy()
        ref_point = ref_p.cpu().numpy()
        self.transform_matrices.append(est_transform.cpu().numpy())
        self.counter += 1 

        # Read the source point cloud
        source_pcd = o3d.io.read_point_cloud(f"/home/tanazzah/GeoTransformer/data/dentskan/ply2/{self.counter}.ply")
        est_np = est_transform.squeeze(0).cpu().numpy()  # Remove the batch dimension
        #est_np = est_transform.cpu().numpy()  # Remove the batch dimension
        # Convert point cloud to numpy array
        src_points = np.asarray(source_pcd.points)
        
        # Debug: print shapes and values
        print("src_points shape:", src_points.shape)
        print("estimated_transform shape:", est_np.shape)
        print("estimated_transform:", est_np)

        # Ensure the estimated_transform is a valid 4x4 matrix
        #assert est_np.shape == (4, 4), "estimated_transform must be a 4x4 matrix"

        # Apply transformation
        rotation = est_np[:3, :3]
        translation = est_np[:3, 3]
        src_points_transformed = np.matmul(src_points, rotation.T) + translation

        # Save transformed point cloud
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(src_points_transformed)
        transformed_pcd.estimate_normals()
        
        coarse_ref_points = output_dict['ref_node_corr_indices'].cpu().numpy()
        coarse_src_points = output_dict['src_node_corr_indices'].cpu().numpy()
        ref_knnpoints = output_dict['ref_node_corr_knn_points'].cpu().numpy()
        src_knnpoints = output_dict['src_node_corr_knn_points'].cpu().numpy()
        src_corr = output_dict['src_corr_points'].cpu().numpy()
        ref_corr = output_dict['ref_corr_points'].cpu().numpy()
        matching = output_dict['matching_scores'].cpu().numpy()

        o3d.io.write_point_cloud(f"/home/tanazzah/GeoTransformer/data/dentskan/ply3/{self.counter}.ply", transformed_pcd)


        
        print("counter",self.counter)    
        rre=0.0
        rte=0.2
        rmse=0.2
        recall=0.2
        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)
        '''with open("transform_matrices.txt", "w") as f:
            for transform_matrix in self.transform_matrices:
                np.savetxt(f, transform_matrix)
                f.write('\n')
        new_matrix=self.transform_matrices[self.counter-1]  
        print("NEW_MATRIX",new_matrix)      
        with open("transform_matrices.txt", "r") as f:
            lines = f.readlines()
        # Update the relevant lines with the new matrix
        start_line = (self.counter-1) * 5
        for i in range(4):
            lines[start_line + i] = " ".join(map(str, new_matrix[i])) + "\n"

        # Write the modified contents back to the file
        with open("transform_matrices.txt", "w") as f:
            f.writelines(lines)
       '''
        #time.sleep(5)
        return {"PIR":0.2, "IR":0.2}
        '''return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }'''



