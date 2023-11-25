import os
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from datasets import load_from_disk, concatenate_datasets
from brainlm_mae.modeling_brainlm import BrainLMForPretraining

if not os.path.exists("inference_plots"):
    os.mkdir("inference_plots")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = BrainLMForPretraining.from_pretrained("pretrained_models/2023-06-06-22_15_00-checkpoint-1400")
print(model.vit.config)
print(model.vit.embeddings.mask_ratio)
print(model.vit.embeddings.config.mask_ratio)
model.vit.embeddings.mask_ratio = 0.0
model.vit.embeddings.config.mask_ratio = 0.0

# Load Data
train_ds = load_from_disk("/home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/train_ukbiobank1000")
print(train_ds)
val_ds = load_from_disk("/home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/val_ukbiobank1000")
print(val_ds)
test_ds = load_from_disk("/home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/test_ukbiobank1000")
print(test_ds)
coords_ds = load_from_disk("/home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/Brain_Region_Coordinates")
print(coords_ds) 

concat_ds = concatenate_datasets([train_ds, val_ds, test_ds])

example0 = concat_ds[0]
# print(example0['Filename'])
# print(example0['Patient ID'])
# print(example0['Order'])
# print(example0['eid'])
# print(example0['Gender'])
# print(example0['Age.At.MHQ'])
# print(example0['Depressed.At.Baseline'])
# print(example0['Neuroticism'])
# print(example0['Self.Harm.Ever'])
# print(example0['Not.Worth.Living'])
# print(example0['PCL.Score'])
# print(example0['GAD7.Severity'])


# Inference
variable_of_interest_col_name = "Age.At.MHQ"
recording_col_name = "Subtract_Mean_Divide_Global_STD_Normalized_Recording"

def preprocess_fmri(examples):
    """
    Preprocessing function for dataset samples. This function is passed into Trainer as
    a preprocessor which takes in one row of the loaded dataset and constructs a model
    input sample according to the arguments which model.forward() expects.

    The reason this function is defined inside on main() function is because we need
    access to arguments such as cell_expression_vector_col_name.
    """
    label = examples[variable_of_interest_col_name][0]
    if math.isnan(label):
        label = -1  # replace nans with -1
    else:
        label = int(label)
    label = torch.tensor(label, dtype=torch.int64)
    signal_vector = examples[recording_col_name][0]
    signal_vector = torch.tensor(signal_vector, dtype=torch.float32)

    # Choose random starting index, take window of moving_window_len points for each region
    start_idx = 0
    end_idx = 490  # 24 patches per voxel, * 424 = 10176 total per sample
    signal_window = signal_vector[start_idx: end_idx, :]  # [moving_window_len, num_voxels]
    signal_window = torch.movedim(signal_window, 0, 1)  # --> [num_voxels, moving_window_len]

    # Append signal values and coords
    window_xyz_list = []
    for brain_region_idx in range(signal_window.shape[0]):
        # window_timepoint_list = torch.arange(0.0, 1.0, 1.0 / num_timepoints_per_voxel)

        # Append voxel coordinates
        xyz = torch.tensor([
            coords_ds[brain_region_idx]["X"],
            coords_ds[brain_region_idx]["Y"],
            coords_ds[brain_region_idx]["Z"]
        ], dtype=torch.float32)
        window_xyz_list.append(xyz)
    window_xyz_list = torch.stack(window_xyz_list)

    # Add in key-value pairs for model inputs which CellLM is expecting in forward() function:
    #  signal_vectors and xyz_vectors
    #  These lists will be stacked into torch Tensors by collate() function (defined above).
    examples["signal_vectors"] = signal_window.unsqueeze(0)
    examples["xyz_vectors"] = window_xyz_list.unsqueeze(0)
    examples["label"] = label
    return examples


def collate_fn(example):
    """
    This function tells the dataloader how to stack a batch of examples from the dataset.
    Need to stack gene expression vectors and maintain same argument names for model inputs
    which CellLM is expecting in forward() function:
        expression_vectors, sampled_gene_indices, and cell_indices
    """
    # These inputs will go to model.forward(), names must match
    return {
        "signal_vectors": example["signal_vectors"],
        "xyz_vectors": example["xyz_vectors"],
        "input_ids": example["signal_vectors"],
        "labels": example["label"]
    }
    

# #--- Forward 1 sample through just the model encoder (model.vit) ---#
# with torch.no_grad():
#     example1 = concat_ds[0]
    
#     # Wrap each value in the key:value pairs into a list (expected by preprocess() and collate())
#     example1[recording_col_name] = [example1[recording_col_name]]
#     example1[variable_of_interest_col_name] = [example1[variable_of_interest_col_name]]

#     processed_example1 = preprocess_fmri(example1)
#     encoder_output = model.vit(
#         signal_vectors=processed_example1["signal_vectors"],
#         xyz_vectors=processed_example1["xyz_vectors"],
#         output_attentions=True,
#         output_hidden_states=True
#     )
    
# print("last_hidden_state:", encoder_output.last_hidden_state.shape)
# # [batch_size, num_genes + 1 CLS token, hidden_dim]

# cls_token = encoder_output.last_hidden_state[:,0,:]
# print(cls_token.shape)


#--- Forward all sample through just the model encoder (model.vit) ---#
all_cls_tokens = []
with torch.no_grad():
    for recording_idx in tqdm(range(concat_ds.num_rows)):
        example1 = concat_ds[recording_idx]

        # Wrap each value in the key:value pairs into a list (expected by preprocess() and collate())
        example1[recording_col_name] = [example1[recording_col_name]]
        example1[variable_of_interest_col_name] = [example1[variable_of_interest_col_name]]

        processed_example1 = preprocess_fmri(example1)
        encoder_output = model.vit(
            signal_vectors=processed_example1["signal_vectors"],
            xyz_vectors=processed_example1["xyz_vectors"],
            output_attentions=True,
            output_hidden_states=True
        )

        cls_token = encoder_output.last_hidden_state[:,0,:]  # torch.Size([1, 256])
        all_cls_tokens.append(cls_token.detach().cpu().numpy())
        
all_cls_tokens = np.concatenate(all_cls_tokens, axis=0)
print(all_cls_tokens.shape)
np.save("inference_plots/all_cls_tokens_{}recordinglength.npy".format(490), all_cls_tokens)

# Save raw recordings as well
all_recordings = []
for recording_idx in tqdm(range(concat_ds.num_rows)):
    example1 = concat_ds[recording_idx]
    recording = np.array(example1[recording_col_name], dtype=np.float32)
    recording = recording[:490].flatten()
    all_recordings.append(recording)

all_recordings = np.stack(all_recordings, axis=0)
print(all_recordings.shape)

np.save("inference_plots/all_{}_490len.npy".format(recording_col_name), all_recordings)

# Reload CLS tokens and raw data if needed
all_cls_tokens = np.load("inference_plots/2023-06-24-00_00_00_checkpoint-3000/all_cls_tokens_490recordinglength.npy")
print(all_cls_tokens.shape)

all_recordings = np.load("inference_plots/all_{}.npy".format(recording_col_name))
print(all_recordings.shape)