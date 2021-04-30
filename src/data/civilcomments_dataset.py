import os
import torch
import pandas as pd
import numpy as np
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset
from transformers import BertTokenizer

class CivilCommentsDataset(ConfounderDataset):
    """
    CivilComments dataset. We only consider the subset of examples with identity annotations.
    Labels are 1 if target_name > 0.5, and 0 otherwise.

    95% of tokens have max_length <= 220, and 99.9% have max_length <= 300
    """

    def __init__(self, args):
        self.dataset_name = 'CivilComments'
        self.data_dir = args.data_dir
        self.target_name = args.target_name
        self.confounder_names = args.confounder_names
        self.model = args.model
        if args.batch_size == 32:
            self.max_length = 128
        elif args.batch_size == 24:
            self.max_length = 220
        elif args.batch_size == 16:
            self.max_length = 300

        assert len(self.confounder_names) == 1
        assert self.model.startswith('bert')

        self.data_dir = os.path.join(
            self.data_dir,
        )
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(
                self.data_dir,
                'all_data_with_identities.csv'),
            index_col=0)

        # Get the y values
        self.y_array = (self.metadata_df[self.target_name].values >= 0.5).astype('long')
        self.n_classes = len(np.unique(self.y_array))

        if self.confounder_names[0] == 'only_label':
            self.n_groups = self.n_classes
            self.group_array = self.y_array
        else:
            self.confounder_array = (self.metadata_df[self.confounder_names[0]].values > 0.5).astype('int')
            self.n_confounders = len(self.confounder_names)

            # Map to groups
            self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
            self.group_array = (self.y_array*(self.n_groups/self.n_classes) + self.confounder_array).astype('int')

        # Extract splits
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        for split in self.split_dict:
            self.metadata_df.loc[self.metadata_df['split'] == split, 'split'] = self.split_dict[split]

        self.split_array = self.metadata_df['split'].values

        # Extract text
        self.text_array = list(self.metadata_df['comment_text'])
        self.tokenizer = BertTokenizer.from_pretrained(self.model)


    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        text = self.text_array[idx]
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length, #220
            return_tensors='pt')
        x = torch.stack(
            (tokens['input_ids'],
             tokens['attention_mask'],
             tokens['token_type_ids']),
            dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1

        return x, y, g

    def group_str(self, group_idx):
        if self.n_groups == self.n_classes:
            y = group_idx
            group_name = f'{self.target_name} = {int(y)}'
        else:
            y = group_idx // (self.n_groups/self.n_classes)
            c = group_idx % (self.n_groups//self.n_classes)
            attr_name = self.confounder_names[0]
            group_name = f'{self.target_name} = {int(y)}, {attr_name} = {int(c)}'
        return group_name
