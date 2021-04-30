import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class CelebADataset(ConfounderDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = 'CelebA'
        self.data_dir = args.data_dir
        self.target_name = args.target_name
        self.confounder_names = args.confounder_names

        # Read in attributes
        self.attrs_df = pd.read_csv(
            os.path.join(self.data_dir, 'list_attr_celeba.csv'))

        self.split_df = pd.read_csv(
            os.path.join(self.data_dir, 'list_eval_partition.csv'))

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.data_dir, 'img_align_celeba')
        self.filename_array = self.attrs_df['image_id'].values
        self.attrs_df = self.attrs_df.drop(labels='image_id', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Read in train/val/test splits
        self.split_array = self.split_df['partition'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.indices_by_group = self._get_group_indices()

        self.features_mat = None
        self.train_transform = get_transform_celebA(self.model)
        self.eval_transform = get_transform_celebA(self.model)

        #filter to make sure all images exist for codalab. affects ~1 image.
        indices_to_remove = []
        for i in range(len(self.filename_array)):
            filename = os.path.join(self.data_dir, self.filename_array[i])
            if not os.path.exists(filename):
                print("Removing filename: ", filename)
                indices_to_remove.append(i)
        print(f"Removed {len(indices_to_remove)} files")
        self.filename_array = np.delete(self.filename_array, indices_to_remove)
        self.split_array = np.delete(self.split_array, indices_to_remove)
        self.group_array = np.delete(self.group_array, indices_to_remove)
        self.y_array = np.delete(self.y_array, indices_to_remove)
        assert len(self.filename_array) == len(self.split_array) == len(self.group_array) == len(self.y_array)

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

def get_transform_celebA(model):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    if model_attributes[model]['target_resolution'] is not None:
        target_resolution = model_attributes[model]['target_resolution']
    else:
        target_resolution = (orig_w, orig_h)

    assert target_resolution is not None
    prelim_celebA_transforms = [transforms.CenterCrop(orig_min_dim), transforms.Resize(target_resolution)]
    tensor_celebA_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_list = prelim_celebA_transforms + tensor_celebA_transforms
    transform = transforms.Compose(transform_list)
    return transform
