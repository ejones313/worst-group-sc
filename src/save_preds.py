import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle
from utils import add_dropout
from collections import defaultdict

def mc_dropout_forward(model, batch, n_samples):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    with torch.set_grad_enabled(False):
        all_outputs = []
        for i in range(n_samples):
            outputs = model(batch)
            outputs =  outputs.detach().cpu().numpy()
            all_outputs.append(outputs)
    model.eval()
    return np.stack(all_outputs, axis = 1)

def mc_dropout_bert_forward(model, input_ids, input_masks, segment_ids, batch_labels, n_samples = 10):
    with torch.set_grad_enabled(False):
        all_outputs = []
        model.train()
        for i in range(n_samples):
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_masks,
                token_type_ids=segment_ids,
                labels=batch_labels)[1].detach().cpu().numpy()
            all_outputs.append(outputs)
    model.eval()
    return np.stack(all_outputs, axis = 1)

def run_loader(model, dataset, batch_size = 16, bert = False, chexpert = False,  
        save_dir = None, mc_dropout = False, n_samples = 10):
    if bert:
        assert batch_size == 16
    if chexpert or bert:
        model.cuda()
    with torch.set_grad_enabled(False):
        maxprob_preds = []
        scores = []
        labels = []
        groups = []
        mc_dropout_scores = []
        size = len(dataset)
        n_batches = size // batch_size
        extra_batch = size % batch_size != 0
        if extra_batch:
            n_batches += 1
        n_correct, total = 0,0
        for batch_id in tqdm(range(n_batches)):
            inputs = []
            if bert:
                masks = []
                sids = []
                batch_labels = []
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size if (not extra_batch or batch_id < n_batches - 1) else size
            for i in range(start, end):
                x, y, g = dataset[i]
                labels.append(y)
                groups.append(g)
                if bert:
                    input_ids = x[:, 0]
                    input_masks = x[:, 1]
                    segment_ids = x[:, 2]
                    inputs.append(input_ids)
                    masks.append(input_masks)
                    sids.append(segment_ids)
                    batch_labels.append(y)
                else:
                    inputs.append(x)
            if chexpert:
                batch = torch.Tensor(np.stack(inputs)).float().cuda()
            else:
                batch = torch.stack(inputs).cuda()
            if bert:
                stacked_input_masks = torch.stack(masks).cuda()
                stacked_segment_ids = torch.stack(sids).cuda()
                batch_labels = torch.Tensor(batch_labels).long().cuda()
                outputs = model(
                    input_ids=batch,
                    attention_mask=stacked_input_masks,
                    token_type_ids=stacked_segment_ids,
                    labels=batch_labels
            )[1] # [1] returns logits
            else:
                outputs = model(batch)
            if mc_dropout:
                if bert:
                    mc_dropout_batch_scores = mc_dropout_bert_forward(model, batch, stacked_input_masks, stacked_segment_ids, batch_labels, n_samples = n_samples)
                else:
                    mc_dropout_batch_scores = mc_dropout_forward(model, batch, n_samples)
                mc_dropout_scores.append(mc_dropout_batch_scores)
            batch_scores = outputs.detach().cpu().numpy()
            scores.append(batch_scores)
            outputs = F.log_softmax(outputs, dim = 1).detach()
            outputs = outputs.cpu().numpy()
            if bert:
                n_correct += (outputs.argmax(axis = 1) == batch_labels.detach().cpu().numpy()).sum()
                total += len(batch_labels)
            maxprob_preds.append(outputs)
        maxprob_preds = np.concatenate(maxprob_preds, axis = 0)
        scores = np.concatenate(scores, axis = 0)
        if mc_dropout:
            mc_dropout_scores = np.concatenate(mc_dropout_scores, axis = 0)
        return maxprob_preds, scores, np.array(labels), np.array(groups), mc_dropout_scores

def save_preds(model, data, args):
    model.eval()
    train_loader = data['train_loader'] 
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']

    #only saving validation and test predictions (to save time)
    datasets = [(val_data, 'val'), (test_data, 'test')]
    for dataset, name in datasets:
        print('')
        print("Processing for split {}: ".format(name))
        info_path = os.path.join(args.log_dir, '{}_preds.pkl'.format(name))
        scores_path = os.path.join(args.log_dir, '{}_scores.pkl'.format(name))
        mc_dropout_scores_path = os.path.join(args.log_dir, '{}_mc_scores.pkl'.format(name))
        save_dir = None
        maxprob_preds, scores, labels, groups, mc_dropout_scores = run_loader(model, dataset, chexpert = args.dataset == 'CheXpert',  
            bert = args.model.startswith('bert'), save_dir = save_dir,  mc_dropout = args.mc_dropout)
        with open(info_path, 'wb') as f:
            pickle.dump((maxprob_preds, labels, groups), f)
        with open(scores_path, 'wb') as f:
            pickle.dump((scores, labels, groups), f)
        if args.mc_dropout:
            with open(mc_dropout_scores_path, 'wb') as f:
                pickle.dump(mc_dropout_scores, f)
