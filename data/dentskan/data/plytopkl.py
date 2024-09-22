import os
import open3d as o3d
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def load_ply_file(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    
    # Assuming that labels are stored in a specific way in your files.
    # You may need to adjust this part depending on how labels are stored.
    # Here, we use the file name as the label for simplicity.
    label = int(0)
    
    return {'points': points, 'normals': normals, 'label': label}

def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def process_folder(input_folder, output_folder):
    ply_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.ply')]
    data = [load_ply_file(file_path) for file_path in ply_files]

    # Split the data into training, testing, and validation sets
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Save the data
    save_pkl(train_data, os.path.join(output_folder, 'train.pkl'))
    save_pkl(test_data, os.path.join(output_folder, 'test.pkl'))
    save_pkl(val_data, os.path.join(output_folder, 'validate.pkl'))

input_folder = 'combined'
output_folder = 'combined2'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_folder(input_folder, output_folder)

