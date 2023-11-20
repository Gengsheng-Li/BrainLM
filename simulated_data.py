# import pandas as pd
# import numpy as np
# import os

# # Parameters for mock data generation
# n_samples = 100  # number of samples in each dataset
# n_timepoints = 490  # number of timepoints per voxel
# n_voxels = 50  # number of voxels (for coordinate dataset)

# # Generate mock training and validation datasets
# # Random fMRI signal values for each timepoint in each sample
# train_data = np.random.rand(n_samples, n_timepoints)
# val_data = np.random.rand(n_samples, n_timepoints)

# # Generate mock coordinates dataset
# # Random X, Y, Z coordinates for each voxel
# coords_data = np.random.rand(n_voxels, 3) * 100  # multiplying by 100 for more realistic coordinates

# # Convert to Pandas DataFrames
# train_df = pd.DataFrame(train_data, columns=[f"Timepoint_{i+1}" for i in range(n_timepoints)])
# val_df = pd.DataFrame(val_data, columns=[f"Timepoint_{i+1}" for i in range(n_timepoints)])
# coords_df = pd.DataFrame(coords_data, columns=["X", "Y", "Z"])

# # File paths
# train_dataset_path = './simulated_data/simulated_train_dataset.csv'
# val_dataset_path = './simulated_data/simulated_val_dataset.csv'
# coords_dataset_path = './simulated_data/simulated_coords_dataset.csv'

# # Save datasets to CSV files
# train_df.to_csv(train_dataset_path, index=False)
# val_df.to_csv(val_dataset_path, index=False)
# coords_df.to_csv(coords_dataset_path, index=False)



# import numpy as np
# from datasets import Dataset

# # 参数用于模拟数据生成
# n_samples = 100  # 数据集中的样本数量
# n_timepoints = 490  # 每个样本的时间点数
# n_voxels = 50  # 坐标数据集中的体素数量

# # 生成模拟的fMRI信号数据
# train_signals = np.random.rand(n_samples, n_timepoints)
# val_signals = np.random.rand(n_samples, n_timepoints)

# # 生成模拟的体素坐标数据
# coords = np.random.rand(n_voxels, 3) * 100  # 乘以100以得到更真实的坐标范围

# # 创建字典来构建数据集
# train_dict = {f"timepoint_{i+1}": train_signals[:, i] for i in range(n_timepoints)}
# val_dict = {f"timepoint_{i+1}": val_signals[:, i] for i in range(n_timepoints)}
# coords_dict = {"X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2]}

# # 将字典转换为datasets.Dataset对象
# train_dataset = Dataset.from_dict(train_dict)
# val_dataset = Dataset.from_dict(val_dict)
# coords_dataset = Dataset.from_dict(coords_dict)

# # 保存数据集到磁盘
# train_dataset_path = './simulated_trn_data'
# val_dataset_path = './simulated_val_data'
# coords_dataset_path = './simulated_coords_data'

# train_dataset.save_to_disk(train_dataset_path)
# val_dataset.save_to_disk(val_dataset_path)
# coords_dataset.save_to_disk(coords_dataset_path)



import numpy as np
from datasets import Dataset

# 参数用于模拟数据生成
n_samples = 100  # 数据集中的样本数量
n_timepoints = 490  # 每个样本的时间点数
n_voxels = 50  # 坐标数据集中的体素数量

# 生成模拟的fMRI信号数据
signals = np.random.rand(n_samples, n_timepoints)

# 生成模拟的体素坐标数据
coords = np.random.rand(n_voxels, 3) * 100  # 乘以100以得到更真实的坐标范围

# 创建字典来构建数据集
train_val_dict = {
    'Voxelwise_RobustScaler_Normalized_Recording': signals,  # 添加预期列
    'variable_of_interest_col_name': np.random.randint(0, 2, n_samples)  # 模拟标签列
}
coords_dict = {'X': coords[:, 0], 'Y': coords[:, 1], 'Z': coords[:, 2]}

# 将字典转换为datasets.Dataset对象
train_dataset = Dataset.from_dict(train_val_dict)
val_dataset = Dataset.from_dict(train_val_dict)  # 使用相同的数据作为验证集
coords_dataset = Dataset.from_dict(coords_dict)

# 保存数据集到磁盘
train_dataset_path = './simulated_trn_data_2'
val_dataset_path = './simulated_val_data_2'
coords_dataset_path = './simulated_coords_data_2'

train_dataset.save_to_disk(train_dataset_path)
val_dataset.save_to_disk(val_dataset_path)
coords_dataset.save_to_disk(coords_dataset_path)