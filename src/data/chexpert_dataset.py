import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class CheXpertDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = 'CheXpert'
        self.data_dir = args.data_dir
        self.target_name = args.target_name
        self.confounder_names = args.confounder_names
        
        local_dir = 'CheXpert-v1.0-small'
        self.data_dir = os.path.join(self.data_dir, local_dir)
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        #Currently only supporting pleural effusion.
        label = 'Pleural Effusion'
        self.y_array = self.metadata_df[label].values.astype('int')
        
        unknown_indices = np.where(self.y_array == -1)[0]
        #For each pathology, replacing unknown with 1 following CheXpert paper 
        self.y_array[unknown_indices] = 1
        self.n_classes = 2

        assert len(self.confounder_names) == 1 #Only support one confounder right now
        self.n_confounders = 1
        confounder = self.confounder_names[0]
        if confounder != 'Support_Devices': raise ValueError(f'Unsupported confounder {confounder}')
        self.confounder_array = self.metadata_df['Support Devices'].values.astype('int')
        #Defaulting uncertain to 1, as in https://arxiv.org/abs/1901.07031
        uncertain_indices = np.where(self.confounder_array == -1)[0]
        self.confounder_array[uncertain_indices] = 1
        self.n_groups = 4
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        self.metadata_df['Path'] = self.metadata_df['Path'].apply(lambda x: x[len('CheXpert-v1.0-small/'):])
        self.filename_array = self.metadata_df['Path'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        # subsample dataset to enforce correlation between support device and the label.
        train_group_array = self.group_array[np.where(self.split_array == 0)[0]]
        train_group_counts = [(train_group_array == g).sum() for g in range(self.n_groups)]
        minority_frac = 1/9 #10% of all examples will be minority
        neg_min_count = min([int(train_group_counts[0] * minority_frac), train_group_counts[1]])
        pos_min_count = min([int(train_group_counts[3] * minority_frac), train_group_counts[2]])
        sizes = [int(neg_min_count / minority_frac), neg_min_count, pos_min_count, int(pos_min_count / minority_frac)] 
        self.explicit_subsample(sizes)

        self.indices_by_group = self._get_group_indices()
        self.train_transform = get_transform_chexpert(
            self.model)
        self.eval_transform = get_transform_chexpert(
            self.model)

def get_transform_chexpert(model):
    target_resolution = model_attributes[model]['target_resolution']
    assert target_resolution is not None

    center_transforms= [
        transforms.Scale(target_resolution),
        transforms.CenterCrop(target_resolution)
    ]
    tensor_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    augmentation_list = center_transforms + tensor_transforms
    transform = transforms.Compose(augmentation_list)
    return transform

