import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models import model_attributes
from save_preds import save_preds
from data.data import supported_datasets, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args, add_dropout
from train import train
from transformers import BertForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=supported_datasets, required=True)
    parser.add_argument('-t', '--target_name', required = True,
            help = "Name of the label to be predicted")
    parser.add_argument('-c', '--confounder_names', nargs='+', required = True,
            help = "Name of the confounder used to define groups")
    parser.add_argument('--data_dir', type = str, required = True,
            help = 'Directory where the data is stored.')
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--model', choices=model_attributes.keys(), default='resnet50')
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--save_preds', action = 'store_true', help = 'Evaluate model with selective classification')
    parser.add_argument('--mc_dropout', action ='store_true', help = 'Save mc dropout scores')
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)
    return parser.parse_args()

def main():
    print("Loading and checking args...")
    args = parse_args()
    check_args(args)
    # BERT-specific configs copied over from run_glue.py
    if args.model.startswith('bert'):
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    #Write for logging; assumes no existing logs.
    mode='w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)
    set_seed(args.seed)

    # Data
    print("Preparing data")
    train_data, val_data, test_data = prepare_data(args, train=True)

    print("Setting up loader")
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    ## Initialize model
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained = True)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
        if args.mc_dropout:
            model = add_dropout(model, 'fc')
    elif args.model == 'densenet121':
        model = torchvision.models.densenet121(pretrained = True)
        d = model.classifier.in_features
        model.classifier = nn.Linear(d, n_classes)
        if args.mc_dropout:
            model = add_dropout(model, 'classifier')
    elif args.model == 'bert-base-uncased':
        print("Loading bert")
        model = BertForSequenceClassification.from_pretrained(
                args.model,
                num_labels = n_classes)
    else:
        raise ValueError('Model not recognized.')

    logger.flush()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    print("Getting loggers")
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode) 
    val_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode) 

    print("Starting to train...")
    train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args, epoch_offset = 0, 
            train = True)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()

    if args.save_preds:
        save_preds(model, data, args)
        return

def check_args(args):
    assert args.confounder_names
    assert args.target_name

if __name__=='__main__':
    main()
