import numpy as np
import pandas as pd
import os
from importlib import reload
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

#CheXpert specific arguments
def load_preds(path):
    if not os.path.exists(path):
        raise ValueError('Invalid path...')
    with open(path, 'rb') as f:
        preds, labels, groups = pickle.load(f)
    return preds, labels, groups

def aggregate_studies(preds, labels, groups, scores, split):
    split2split_name = {'val': 1, 'test': 2}
    data = pd.read_csv('CheXpert-device/chexpert_paths.csv')
    data = data.loc[data['split'] == split2split_name[split]]
    data['study'] = data['Path'].apply(lambda x: '{}_{}'.format(x.split('/')[2], x.split('/')[3]))
    study_ids = data['study'].values
    studies = set(study_ids)
    new_labels = []
    new_preds = []
    new_scores = []
    new_groups = []
    new_indices = []
    #One example per study
    for study in studies:
        ids = np.where(study_ids == study)[0]
        study_labels = labels[ids]
        study_groups = groups[ids]
        study_preds = preds[ids]
        study_scores = scores[ids]
        rel_idx = ids[study_preds[:,1].argmax()]
        new_indices.append(rel_idx)
        new_labels.append(study_labels[0])
        #Take the most positive case...
        new_preds.append(study_preds[study_preds[:,1].argmax()])
        new_scores.append(study_scores[study_scores[:,1].argmax()])
        new_groups.append(study_groups[0])
    return np.array(new_preds), np.array(new_labels), np.array(new_groups), np.array(new_scores), np.array(new_indices)

def accuracy(labels, preds, correct, shift_weights):
    if len(correct) == 0:
        return -1
    #weight different points 
    weighted_correct = correct.astype(int) * shift_weights
    return weighted_correct.sum()/shift_weights.sum()

def robinhood_update(group_correct_counts, group_incorrect_counts, current_accuracies, inc_to_remove, cor_to_remove):
    #While there are examples remaining
    n_groups = len(current_accuracies)
    group_total_counts = group_correct_counts + group_incorrect_counts
    sorted_indices = np.argsort(current_accuracies)

    #allocate to worst-groups
    n_worst_groups = 1
    for i in range(1, n_groups):
        if current_accuracies[sorted_indices[i]] - current_accuracies[sorted_indices[i - 1]] > 1e-3:
            break
        n_worst_groups += 1
        if n_worst_groups == n_groups:
            break
    
    if n_worst_groups == n_groups:
        group_incorrect_counts -= inc_to_remove * group_incorrect_counts/group_incorrect_counts.sum()
        group_total_counts = group_correct_counts + group_incorrect_counts
        current_accuracies = group_correct_counts/group_total_counts
    else:
        while inc_to_remove > 1e-4:
            current_wg_acc = current_accuracies[sorted_indices[0]]
            #Compute the number of incorrect removals required to get to the next group
            if n_worst_groups == n_groups:
                req_inc_to_remove = inc_to_remove
            else:
                acc_diff = current_accuracies[sorted_indices[n_worst_groups]] - current_wg_acc
                req_inc_to_remove = acc_diff * np.array([group_total_counts[sorted_indices[i]] for i in range(n_worst_groups)]).sum()
            inc_actually_removed = min([req_inc_to_remove, inc_to_remove])
            inc_to_remove -= inc_actually_removed
            rel_inc_counts = np.array([group_incorrect_counts[sorted_indices[i]] for i in range(n_worst_groups)])
            inc_fracs = rel_inc_counts/rel_inc_counts.sum()
            for i in range(n_worst_groups):
                group_incorrect_counts[sorted_indices[i]] -= inc_fracs[i] * inc_actually_removed
            group_total_counts = group_correct_counts + group_incorrect_counts
            current_accuracies = group_correct_counts/group_total_counts
            n_worst_groups += 1

    n_best_groups = 1
    sorted_indices = np.argsort(current_accuracies)[::-1]
    for i in range(1, n_groups):
        if current_accuracies[sorted_indices[i]] - current_accuracies[sorted_indices[i - 1]] < 1e-3:
            break
        n_best_groups += 1
        
    if n_best_groups == n_groups:
        group_correct_counts -= cor_to_remove * group_correct_counts / group_correct_counts.sum()
        group_total_counts = group_correct_counts + group_incorrect_counts
        current_accuracies = group_correct_counts/group_total_counts
    else:
        while cor_to_remove > 1e-4:
            current_bg_acc = current_accuracies[sorted_indices[0]]
            if n_best_groups == n_groups:
                req_cor_to_remove = cor_to_remove
            else:
                acc_diff = current_bg_acc - current_accuracies[sorted_indices[n_best_groups]]
                req_cor_to_remove = acc_diff * np.array([group_total_counts[sorted_indices[i]] for i in range(n_best_groups)]).sum()
            cor_actually_removed = min([req_cor_to_remove, cor_to_remove])
            cor_to_remove -= cor_actually_removed
            rel_cor_counts = np.array([group_correct_counts[sorted_indices[i]] for i in range(n_best_groups)])
            cor_fracs = rel_cor_counts/rel_cor_counts.sum()
            for i in range(n_best_groups):
                group_correct_counts[sorted_indices[i]] -= cor_fracs[i] * cor_actually_removed
            group_total_counts = group_correct_counts + group_incorrect_counts
            current_accuracies = group_correct_counts/group_total_counts
            n_best_groups += 1
        
    return group_correct_counts, group_incorrect_counts, group_total_counts, current_accuracies
    

def get_robinhood_selective_accuracy(avg_curve, coverages, is_correct, groups, shift_weights):
    #Policy: if correct example, abstain from best group
    n_groups = groups.max() + 1
    n_points = len(is_correct)
    weighted_correct = is_correct.astype(int) * shift_weights
    group_indices = [np.where(groups == g)[0] for g in range(n_groups)]
    group_correct_counts = np.array([weighted_correct[group_indices[g]].sum() for g in range(n_groups)])
    group_total_counts = np.array([shift_weights[group_indices[g]].sum() for g in range(n_groups)])
    group_incorrect_counts = group_total_counts - group_correct_counts
    assert coverages[0] > 0.1 #Sanity, doesn't have to be 1 anymore
    
    current_accuracies = group_correct_counts / group_total_counts
    robinhood_accs = defaultdict(list)
    #This is bad, but sticking with existing format
    for g in range(n_groups):
        robinhood_accs[g].append(current_accuracies[g])
    
    #New curves
    for i in range(1, len(coverages)):
        curr_predicted = n_points * coverages[i]
        curr_correct = avg_curve[i] * curr_predicted
        curr_incorrect = curr_predicted - curr_correct  
        
        prev_predicted = n_points * coverages[i - 1]
        prev_correct = avg_curve[i - 1] * prev_predicted
        prev_incorrect = prev_predicted - prev_correct 
        
        inc_to_remove = prev_incorrect - curr_incorrect
        cor_to_remove = prev_correct - curr_correct
        group_correct_counts, group_incorrect_counts, group_total_counts, current_accuracies = robinhood_update(
                group_correct_counts, group_incorrect_counts, current_accuracies, inc_to_remove, cor_to_remove)  
        for g in range(n_groups):
            robinhood_accs[g].append(current_accuracies[g])
    return robinhood_accs

def get_gab_curves(avg_curve, coverages, is_correct, groups, shift_weights):
    #Get curves for the baseline
    n_groups = groups.max() + 1
    n_points = len(is_correct)
    group_indices = [np.where(groups == g)[0] for g in range(n_groups)]
    incorrect_counts = []
    correct_counts = []
    for g in range(n_groups):
        group_correct = is_correct[group_indices[g]]
        group_weights = shift_weights[group_indices[g]]
        assert group_weights.max() - group_weights.mean() < 1e-5
        incorrect_counts.append((group_correct == 0).sum() * group_weights[0])
        correct_counts.append((group_correct == 1).sum() * group_weights[0])
        
    incorrect_counts, correct_counts = np.array(incorrect_counts), np.array(correct_counts)
    total_counts = incorrect_counts + correct_counts
    incorrect_fracs, correct_fracs = incorrect_counts/incorrect_counts.sum(), correct_counts/correct_counts.sum()
    
    group2curve = defaultdict(list)
    group2abstain= defaultdict(list)
    for i in range(len(avg_curve)):
        coverage = coverages[i]
        total_predicted = int(n_points * coverage)
        total_correct = avg_curve[i] * total_predicted
        total_incorrect = total_predicted - total_correct
        for g in range(n_groups):
            group_correct = total_correct * correct_fracs[g]
            group_incorrect = total_incorrect * incorrect_fracs[g]
            group_acc = group_correct / (group_incorrect + group_correct)
            group2curve[g].append(group_acc)
            group2abstain[g].append(1 - (group_correct + group_incorrect)/total_counts[g])
    group2abstain[-1] = (1 - coverage)
    group2curve[-1] = avg_curve
    return group2curve, group2abstain 

def get_group_curves(is_correct, confidences, preds, labels, groups, n_points = 200, shift = None):
    #Get group curves and abstentions 
    n_groups = groups.max() + 1
    if shift is None:
        shift_weights = np.ones(groups.shape[0])
    else:
        shift_weights = get_shift_weights(groups, shift)
    #Groups are consecutive nonnegative indices starting at 0
    assert len(set(groups)) == n_groups and groups.min() == 0
    group_counts = [(groups == g).sum() for g in range(n_groups)]
    num_examples = len(is_correct)
    #Remove indices from the back first, since those are lowest confidence
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_correct = is_correct[sorted_indices]
    sorted_preds = preds[sorted_indices]
    sorted_labels = labels[sorted_indices]
    sorted_groups = groups[sorted_indices]
    sorted_shift_weights = shift_weights[sorted_indices]
    #First examples are the most confident, last to predict
    cumulative_coverages = np.array([sorted_shift_weights[:i].sum() for i in range(len(sorted_shift_weights))]) / sorted_shift_weights.sum()
    
    curves = defaultdict(list)
    abstentions = defaultdict(list)
    coverages = np.array([1 - i/n_points for i in range(n_points)])
    new_coverages = []
    for i in range(len(coverages)):
        possible_indices = np.where(cumulative_coverages > coverages[i])[0]
        if len(possible_indices) == 0:
            n_predictions = len(sorted_correct) - 1
        else:
            n_predictions = possible_indices[0] - 1
            if n_predictions < 0:
                n_predictions = 0
        new_coverages.append(cumulative_coverages[n_predictions])
        rel_correct = sorted_correct[:n_predictions]
        rel_preds = sorted_preds[:n_predictions]
        rel_groups = sorted_groups[:n_predictions]
        rel_labels = sorted_labels[:n_predictions]
        rel_shift_weights = sorted_shift_weights[:n_predictions]
        curves[-1].append(accuracy(rel_labels, rel_preds, rel_correct, rel_shift_weights))
        abstentions[-1].append(1 - cumulative_coverages[n_predictions])
        for g in range(n_groups):
            rel_group_indices = np.where(rel_groups == g)[0]
            rel_group_preds = rel_preds[rel_group_indices]
            rel_group_labels = rel_labels[rel_group_indices]
            rel_group_correct = rel_correct[rel_group_indices]
            rel_group_shift_weights = rel_shift_weights[rel_group_indices]
            curves[g].append(accuracy(rel_group_labels, rel_group_preds, rel_group_correct, rel_group_shift_weights))
            abstentions[g].append(1 - len(rel_group_indices)/group_counts[g])     
    for g in curves:
        curves[g] = np.array(curves[g])
        abstentions[g] = np.array(abstentions[g])
    new_coverages = np.array(new_coverages)
    return curves, new_coverages, abstentions, is_correct

def get_shift_weights(groups, shift):
    n_groups = groups.max() + 1
    group_indices = [np.where(groups == g)[0] for g in range(n_groups)]
    group_counts = np.array([len(group_indices[g]) for g in range(n_groups)])
    new_group_weights = shift / (group_counts / group_counts.sum())
    new_weights = np.array([new_group_weights[groups[i]] for i in range(groups.shape[0])])
    return new_weights

def get_mc_dropout_confidences(bin_preds, mc_scores):
    mc_probs = np.zeros(shape = mc_scores.shape)
    n_batches, n_samples, n_classes = mc_scores.shape
    for i in tqdm(range(n_batches)):
        mc_probs[i] = mc_scores[i] - mc_scores[i].mean(axis = 1).reshape(-1, 1)
        mc_probs[i] = np.exp(mc_probs[i])/np.exp(mc_probs[i]).sum(axis = 1)[:, np.newaxis]
        rel_mc_probs = mc_probs[np.arange(mc_scores.shape[0]), :, bin_preds]
    mc_stds = rel_mc_probs.std(axis = 1)
    confidences = -mc_stds
    return confidences
