import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
import random
from collections import defaultdict

class ConfounderDataset(Dataset):
    def __init__(self, args):
        self.model = args.model

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        img_filename = os.path.join(
            self.data_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.eval_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
          self.eval_transform):
            img = self.eval_transform(img)
        x = img
        return x,y,g

    def _get_group_indices(self):
        group2indices = {}
        for g in range(self.n_groups):
            group_indices = np.where((self.group_array == g) & (self.split_array == self.split_dict['train']))[0]
            group2indices[g] = group_indices
        return group2indices

    def explicit_subsample(self, sizes):
        # We use this for support device confounder
        assert len(sizes) == self.n_groups
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        group_counts = np.array([(self.group_array[train_indices] == group_idx).sum() for group_idx in range(self.n_groups)])
        for i in range(self.n_groups):
            final_num_points = min([sizes[i], group_counts[i]])
            self.undersample(i, final_num_points)

    def undersample(self, undersample_group, new_size):
        # We use this for support device confounder
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        current_undersample_count  = (self.group_array[train_indices] == undersample_group).sum()
        if new_size > current_undersample_count:
            raise ValueError("Currently exist {} train examples in group {}, so cannot reduce to {}".format(current_undersample_count,
                undersample_group, new_size))
        undersample_indices = np.where((self.split_array == self.split_dict['train']) & (self.group_array == undersample_group))[0]
        undersample_delete_indices = np.random.choice(undersample_indices, current_undersample_count - new_size, replace = False) 
        self.y_array = np.delete(self.y_array, undersample_delete_indices)
        self.group_array = np.delete(self.group_array, undersample_delete_indices)
        self.filename_array = np.delete(self.filename_array, undersample_delete_indices)
        self.split_array = np.delete(self.split_array, undersample_delete_indices)

    def get_splits(self, splits):
        subsets = {}
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            subsets[split] = Subset(self, indices)
            subsets[split].y_array = self.y_array[indices]
            subsets[split].group_array = self.group_array[indices]
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        assert len(self.confounder_names) == 1
        return group_name
