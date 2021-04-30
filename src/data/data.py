import torch
import numpy as np
from torch.utils.data import Subset
from data.confounder_utils import prepare_confounder_data

supported_datasets = ['CelebA', 'CivilComments', 'Waterbirds', 'CheXpert', 'MultiNLI']

def prepare_data(args, train, return_full_dataset=False):
    return prepare_confounder_data(args, train, return_full_dataset)

def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')
