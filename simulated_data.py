import numpy as np
from datasets import Dataset


n_subjects = 100  # 数据集中的样本数量
n_timepoints = 490  # 每个样本的时间点数
n_voxels = 424 # 坐标数据集中的体素数量

recordings = np.random.rand(n_subjects, n_timepoints, n_voxels)
# recordings = np.random.rand(n_subjects, n_voxels, n_timepoints)
labels = np.random.randint(1, 100, n_subjects)
# coords = np.random.rand(n_subjects, n_voxels, 3) * 100  # 乘以100以得到更真实的坐标范围
index =  np.arange(1, 425)
coords = np.random.rand(424, 3) * 100

train_val_dict = {
    'Voxelwise_RobustScaler_Normalized_Recording': recordings, # All_Patient_All_Voxel_Normalized_Recording
    'Age.At.MHQ': labels # Desired label for each patient
}
coords_dict = {'Index': index, 
               'X': coords[ :, 0], 
               'Y': coords[ :, 1], 
               'Z': coords[ :, 2]}
# coords_dict = {'X': coords[:, :, 0].flatten(), 'Y': coords[:, :, 1].flatten(), 'Z': coords[:, :, 2].flatten()}


train_dataset = Dataset.from_dict(train_val_dict)
val_dataset = Dataset.from_dict(train_val_dict)  # 使用相同的数据作为验证集
coords_dataset = Dataset.from_dict(coords_dict)


train_dataset_path = './simulated_trn_data_2'
val_dataset_path = './simulated_val_data_2'
coords_dataset_path = './simulated_coords_data_2'

train_dataset.save_to_disk(train_dataset_path)
val_dataset.save_to_disk(val_dataset_path)
coords_dataset.save_to_disk(coords_dataset_path)