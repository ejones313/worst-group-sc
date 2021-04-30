import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class WaterbirdsDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = 'Waterbirds'
        self.data_dir = args.data_dir
        self.target_name = args.target_name
        self.confounder_names = args.confounder_names

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.indices_by_group = self._get_group_indices()
        self.features_mat = None
        self.train_transform = get_transform_cub(self.model)
        self.eval_transform = get_transform_cub(self.model)

def get_transform_cub(model):
    scale = 256.0/224.0
    target_resolution = model_attributes[model]['target_resolution']
    assert target_resolution is not None
    center_transforms= [
        transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
        transforms.CenterCrop(target_resolution)
    ]
    tensor_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    augmentation_list = center_transforms + tensor_transforms
    transform = transforms.Compose(augmentation_list)
    return transform
