import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.dro_dataset import DRODataset
from data.celebA_dataset import CelebADataset
from data.waterbirds_dataset import WaterbirdsDataset
from data.multinli_dataset import MultiNLIDataset
from data.chexpert_dataset import CheXpertDataset
from data.civilcomments_dataset import CivilCommentsDataset

confounder_settings = {
    'CelebA':{
        'constructor': CelebADataset
    },
    'CivilComments':{
        'constructor': CivilCommentsDataset 
    },
    'Waterbirds':{
        'constructor': WaterbirdsDataset
    },
    'CheXpert':{
        'constructor': CheXpertDataset
    },
    'MultiNLI':{
        'constructor': MultiNLIDataset
    }
}

def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](args=args)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        if args.dataset == 'CheXpert':
            splits = ['train', 'val']
        else:
            splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets
