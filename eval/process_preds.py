from eval_utils import *
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from PIL import Image
from matplotlib.ticker import FormatStrFormatter

#Shift reweights so the validation distribution over groups matches train
dataset2shift = {
        'CelebA': None, 
        'CivilComments': None, 
        'Waterbirds': np.array([3498,184,56,1057])/np.array([3498,184,56,1057]).sum(),
        'CheXpert-device': np.array([51687,5743,5467,49203])/np.array([51687,5743,5467,49203]).sum(),
        'MultiNLI': None
        }
dataset2erm_worst_group = {'Waterbirds': 2, 'CelebA': 3, 'MultiNLI': 5, 'CheXpert-device': 2, 'CivilComments': 3}
dataset2dro_worst_group = {'Waterbirds': 2, 'CelebA': 3, 'MultiNLI': 5, 'CheXpert-device': 0, 'CivilComments': 1}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs = '+', choices = ['CelebA', 'CivilComments', 'Waterbirds', 'CheXpert-device', 'MultiNLI'])
    parser.add_argument('--opt', choices = ['ERM', 'DRO'])
    parser.add_argument('--mc_dropout', action = 'store_true')
    return parser.parse_args()

def get_bundle_name(args, dataset):
    if args.mc_dropout:
        return f'{dataset}-MC'
    return f'{dataset}-{args.opt}'

def load_output(bundle_name, split = 'test'):
    assert split in ['val', 'test']
    preds_dir = os.path.join(bundle_name, 'preds')
    if not os.path.exists(preds_dir):
        raise ValueError(f'Directory {preds_dir} does not exist or does not have predictions (possibly an error running the bundle?')
    with open(os.path.join(preds_dir, f'{split}_preds.pkl'), 'rb') as f:
        preds, labels, groups = pickle.load(f)
    with open(os.path.join(preds_dir, f'{split}_scores.pkl'), 'rb') as f:
        scores, score_labels, score_groups = pickle.load(f)
        #normalize scores to 0 mean
        scores = scores - scores.mean(axis = 1).reshape(-1,1)
    #make sure things line up
    assert (score_labels == labels).all()
    assert (score_groups == groups).all()
    mc_path = os.path.join(preds_dir, f'{split}_mc_scores.pkl')
    if os.path.exists(mc_path):
        with open(mc_path, 'rb') as f:
            mc_dropout_scores = pickle.load(f)
    else:
        mc_dropout_scores = None

    #need to combine x-rays from the same study (different views) for CheXpert
    if 'CheXpert-device' in bundle_name:
        preds, labels, groups, scores, indices = aggregate_studies(preds, labels, groups, scores, 'test')
        if mc_dropout_scores is not None:
            mc_dropout_scores = mc_dropout_scores[indices]

    output_dict = {
            'preds': preds,
            'labels': labels,
            'groups': groups,
            'scores': scores,
            'mc_scores': mc_dropout_scores
            }
    return output_dict

def get_plot_data(preds, labels, groups, n_points = 200, shift = None):
    is_correct = preds.argmax(axis = 1) == labels
    confidences = preds.max(axis = 1)
    shift_weights = np.ones(groups.shape[0])
    if shift is not None:
        shift_weights = get_shift_weights(groups, shift)
    #average selective accuracy is indexed by -1
    #abstain rates are available since they differe based on group.
    selective_accuracy, coverages, abstentions, is_correct = get_group_curves(is_correct, confidences, preds, labels, groups,
            shift = shift)
    baseline_selective_accuracy, baseline_abstentions = get_gab_curves(selective_accuracy[-1], coverages, is_correct, groups, shift_weights)
    robinhood_selective_accuracy = get_robinhood_selective_accuracy(selective_accuracy[-1], coverages, is_correct, groups, shift_weights)
    plot_data = {'selective_accuracy': selective_accuracy, 'abstain_rate': abstentions, 
            'gab_selective_accuracy': baseline_selective_accuracy, 'gab_abstain_rate': baseline_abstentions, 
                'robinhood_selective_accuracy': robinhood_selective_accuracy, 'shift_weights': shift_weights,
                'coverages': coverages}
    return plot_data

def postprocess_scores(output_dict, shift = None):
    #use scores or predictions depending on multiclass (normalizing to 0 mean doesn't necessarily give the right ordering taking the max for scores)
    preds, labels, groups, scores, mc_dropout_scores = [output_dict[key] for key in ['preds', 'labels', 'groups', 'scores', 'mc_scores']]
    if labels.max() > 1:
        plot_data = get_plot_data(preds, labels, groups, shift = shift)
    else:
        plot_data = get_plot_data(scores, labels, groups, shift = shift)
    if mc_dropout_scores is not None:
        is_correct = scores.argmax(axis = 1) == labels
        print("Getting dropout confidences...")
        mc_dropout_confidences = get_mc_dropout_confidences(preds.argmax(axis = 1), mc_dropout_scores)
        print("Done!")
        mc_dropout_selective_accuracy, mc_marginal_coverages, mc_abstentions, is_correct = get_group_curves(
            is_correct, mc_dropout_confidences, preds, labels, groups, shift = shift)
        plot_data['mc_selective_accuracy'] = mc_dropout_selective_accuracy
        plot_data['mc_abstain_rate'] = mc_abstentions
        plot_data['mc_coverages'] = mc_marginal_coverages
        plot_data['mc_confidences'] = mc_dropout_confidences
    for key in output_dict: plot_data[key] = output_dict[key]
    return plot_data


def main():
    np.random.seed(0)
    args = parse_args()
    assert len(args.datasets) > 1
    dataset2plot_data = {}
    for dataset in args.datasets:
        bundle_name = get_bundle_name(args, dataset)
        output_dict = load_output(bundle_name)
        shift = dataset2shift[dataset]
        plot_data = postprocess_scores(output_dict, shift = shift)
        plot_data['n_groups'] = 4 if dataset != 'MultiNLI' else 6
        plot_data['worst_group'] = dataset2erm_worst_group[dataset] if args.opt == 'ERM' else dataset2dro_worst_group[dataset]
        dataset2plot_data[dataset] = plot_data
    print("Generated plot data!")

    if not os.path.exists('figures'):
        os.mkdir('figures')
    if args.mc_dropout:
        mc_dropout_fig(args.datasets, dataset2plot_data)
    else:
        if args.opt == 'ERM':
            acc_coverage_fig_one_group(args.datasets, dataset2plot_data, args.opt)
            acc_coverage_fig(args.datasets, dataset2plot_data, args.opt)
        gab_baseline_fig(args.datasets, dataset2plot_data, args.opt)
        margin_plot(args.datasets, dataset2plot_data, args.opt)

def acc_coverage_fig(ordered_datasets, dataset2plot_data, opt):
    assert len(ordered_datasets) > 1
    for dataset in ordered_datasets:
        assert dataset in dataset2plot_data
    ordered_keys = ['accuracy', 'abstain_rate']
    plt.clf()
    group_dict = {i:i for i in range(-1,6)}
    fig, axs = plt.subplots(2, len(ordered_datasets), figsize=(15, 6), sharey = True if opt == 'ERM' else False, sharex = True)
    for i, dataset in enumerate(ordered_datasets):
        plot_data = dataset2plot_data[dataset]
        n_groups = plot_data['n_groups']
        worst_group = plot_data['worst_group']
        accuracies = plot_data['selective_accuracy']
        abstain_rates = plot_data['abstain_rate']
        axs[0,i].set_title(dataset, fontsize = 17)
        groups = [n_groups - 1, -1] + [i for i in range(n_groups - 1)] + [-1]
        for group in groups:
            group_y = np.array(plot_data['selective_accuracy'][group])
            #filter out indices where nothing
            rel_indices = np.where(group_y != -1)[0]
            color = '0.75'
            if group == worst_group:
                color = 'firebrick'
            elif group == -1:
                color = 'darkblue'
            axs[0,i].plot(plot_data['coverages'][rel_indices], group_y[rel_indices], color = color)
            axs[0, i].tick_params(axis='both', labelsize=16)
            axs[0, i].set_xticks(np.array([0,0.25, 0.5, 0.75, 1]))
            if opt == 'DRO':
                axs[0, i].set_ylim(0.6, 1.01)
                axs[0, i].set_yticks(np.array([0.6, 0.7, 0.8, 0.9, 1]))
                axs[0, i].tick_params(labelbottom=False) 
                if i != 0:
                    axs[0, i].tick_params(labelleft=False)
                    axs[1, i].tick_params(labelleft = False)
            else:
                axs[0, i].set_yticks(np.array([0,0.25, 0.5, 0.75, 1]))
            axs[0, i].grid(b = True, linestyle = '--', color = '0.8')
        if dataset == 'MultiNLI' and opt == 'ERM':
            axs[0,i].legend(['Worst group', 'Average', 'Other groups'], fontsize = 13, loc = 'lower right')
        elif dataset == 'CelebA' and opt == 'DRO':
            axs[0,i].legend(['Worst group', 'Average', 'Other groups'], fontsize = 13, loc = 'lower left')
        
        for group in groups:
            group_y = np.array(plot_data['abstain_rate'][group])
            rel_indices = np.where(group_y != -1)[0]
            color = '0.75'
            if group == worst_group:
                color = 'firebrick'
            elif group == -1:
                color = 'darkblue'
            axs[1,i].plot(plot_data['coverages'][rel_indices], 1 - group_y[rel_indices], color = color)
            axs[1, i].tick_params(axis='both', labelsize=16, length = 2)
            
            axs[1, i].set_xticks(np.array([0, 0.25, 0.5, 0.75, 1]))
            axs[1, i].set_xticklabels([0, None, 0.5, None, 1])
            axs[1, i].set_yticks(np.array([0,0.25, 0.5, 0.75, 1]))

            axs[1, i].yaxis.set_major_formatter(FormatStrFormatter('%g'))
            axs[1, i].grid(b = True, linestyle = '--', color = '0.8')


    axs[0,0].set_ylabel('Group sel. accuracy', fontsize = 17)
    axs[1,0].set_ylabel('Group coverage', fontsize = 17)
    for i in range(len(ordered_datasets)):
        axs[1,i].set_xlabel('Average coverage', fontsize = 17)
    fig.subplots_adjust(wspace=0.25, hspace=0.1)
    fig.savefig(f'figures/sc_{opt}_results.pdf', bbox_inches = 'tight')
    print("Finished main figure!")

def acc_coverage_fig_one_group(ordered_datasets, dataset2plot_data, opt):
    assert len(ordered_datasets) > 1
    for dataset in ordered_datasets:
        assert dataset in dataset2plot_data
    ordered_keys = ['accuracy', 'abstain_rate']
    plt.clf()
    group_dict = {i:i for i in range(-1,6)}
    fig, axs = plt.subplots(1, len(ordered_datasets), figsize=(15, 3.5), sharey = True if opt == 'ERM' else False)
    for i, dataset in enumerate(ordered_datasets):
        plot_data = dataset2plot_data[dataset]
        n_groups = plot_data['n_groups']
        worst_group = plot_data['worst_group']
        accuracies = plot_data['selective_accuracy']
        abstain_rates = plot_data['abstain_rate']
        axs[i].set_title(dataset, fontsize = 17)
        groups = [n_groups - 1, -1] + [i for i in range(n_groups - 1)] + [-1]
        for group in groups:
            group_y = np.array(plot_data['selective_accuracy'][group])
            #filter out indices where nothing
            rel_indices = np.where(group_y != -1)[0]
            color = '0.75'
            if group == worst_group:
                color = 'firebrick'
            elif group == -1:
                color = 'darkblue'
            axs[i].plot(plot_data['coverages'][rel_indices], group_y[rel_indices], color = color)
            axs[i].tick_params(axis='both', labelsize=16)
            axs[i].set_xticks(np.array([0,0.25, 0.5, 0.75, 1]))
            if opt == 'DRO':
                axs[i].set_ylim(0.6, 1.01)
                axs[i].set_yticks(np.array([0.6, 0.7, 0.8, 0.9, 1]))
                axs[i].tick_params(labelbottom=False) 
                if i != 0:
                    axs[i].tick_params(labelleft=False)
                    axs[i].tick_params(labelleft = False)
            else:
                axs[i].set_yticks(np.array([0,0.25, 0.5, 0.75, 1]))
            axs[i].grid(b = True, linestyle = '--', color = '0.8')
        if dataset == 'MultiNLI' and opt == 'ERM':
            axs[i].legend(['Worst group', 'Average', 'Other groups'], fontsize = 13, loc = 'lower right')
        elif dataset == 'CelebA' and opt == 'DRO':
            axs[i].legend(['Worst group', 'Average', 'Other groups'], fontsize = 13, loc = 'lower left')
        
        for group in groups:
            group_y = np.array(plot_data['abstain_rate'][group])
            rel_indices = np.where(group_y != -1)[0]
            color = '0.75'
            if group == worst_group:
                color = 'firebrick'
            elif group == -1:
                color = 'darkblue'
            #axs[i].tick_params(axis='both', labelsize=16, length = 2)
            axs[i].set_xticks(np.array([0, 0.25, 0.5, 0.75, 1]))
            axs[i].set_xticklabels([0, None, 0.5, None, 1])


    axs[0].set_ylabel('Group sel. accuracy', fontsize = 17)
    for i in range(len(ordered_datasets)):
        axs[i].set_xlabel('Average coverage', fontsize = 17)
    fig.subplots_adjust(wspace=0.25, hspace=0.1)
    fig.savefig(f'figures/sc_{opt}_one_group_results.pdf', bbox_inches = 'tight')
    print("Finished single_group figure!")

def acc_coverage_no_abstain(ordered_datasets, dataset2plot_data, opt):
    assert len(ordered_datasets) > 1
    for dataset in ordered_datasets:
        assert dataset in dataset2plot_data
    ordered_keys = ['accuracy', 'abstain_rate']
    plt.clf()
    group_dict = {i:i for i in range(-1,6)}
    fig, axs = plt.subplots(2, len(ordered_datasets), figsize=(15, 6), sharey = True if opt == 'ERM' else False, sharex = True)
    for i, dataset in enumerate(ordered_datasets):
        plot_data = dataset2plot_data[dataset]
        n_groups = plot_data['n_groups']
        worst_group = plot_data['worst_group']
        accuracies = plot_data['selective_accuracy']
        abstain_rates = plot_data['abstain_rate']
        axs[0,i].set_title(dataset, fontsize = 17)
        groups = [n_groups - 1, -1] + [i for i in range(n_groups - 1)] + [-1]
        for group in groups:
            group_y = np.array(plot_data['selective_accuracy'][group])
            #filter out indices where nothing
            rel_indices = np.where(group_y != -1)[0]
            color = '0.75'
            if group == worst_group:
                color = 'firebrick'
            elif group == -1:
                color = 'darkblue'
            axs[0,i].plot(plot_data['coverages'][rel_indices], group_y[rel_indices], color = color)
            axs[0, i].tick_params(axis='both', labelsize=16)
            axs[0, i].set_xticks(np.array([0,0.25, 0.5, 0.75, 1]))
            if opt == 'DRO':
                axs[0, i].set_ylim(0.6, 1.01)
                axs[0, i].set_yticks(np.array([0.6, 0.7, 0.8, 0.9, 1]))
                axs[0, i].tick_params(labelbottom=False) 
                if i != 0:
                    axs[0, i].tick_params(labelleft=False)
                    axs[1, i].tick_params(labelleft = False)
            else:
                axs[0, i].set_yticks(np.array([0,0.25, 0.5, 0.75, 1]))
            axs[0, i].grid(b = True, linestyle = '--', color = '0.8')
        if dataset == 'MultiNLI' and opt == 'ERM':
            axs[0,i].legend(['Worst group', 'Average', 'Other groups'], fontsize = 13, loc = 'lower right')
        elif dataset == 'CelebA' and opt == 'DRO':
            axs[0,i].legend(['Worst group', 'Average', 'Other groups'], fontsize = 13, loc = 'lower left')
        
        for group in groups:
            group_y = np.array(plot_data['abstain_rate'][group])
            rel_indices = np.where(group_y != -1)[0]
            color = '0.75'
            if group == worst_group:
                color = 'firebrick'
            elif group == -1:
                color = 'darkblue'
            axs[1,i].plot(plot_data['coverages'][rel_indices], 1 - group_y[rel_indices], color = color)
            axs[1, i].tick_params(axis='both', labelsize=16, length = 2)
            
            axs[1, i].set_xticks(np.array([0, 0.25, 0.5, 0.75, 1]))
            axs[1, i].set_xticklabels([0, None, 0.5, None, 1])
            axs[1, i].set_yticks(np.array([0,0.25, 0.5, 0.75, 1]))

            axs[1, i].yaxis.set_major_formatter(FormatStrFormatter('%g'))
            axs[1, i].grid(b = True, linestyle = '--', color = '0.8')


    axs[0,0].set_ylabel('Group sel. accuracy', fontsize = 17)
    axs[1,0].set_ylabel('Group coverage', fontsize = 17)
    for i in range(len(ordered_datasets)):
        axs[1,i].set_xlabel('Average coverage', fontsize = 17)
    fig.subplots_adjust(wspace=0.25, hspace=0.1)
    fig.savefig(f'figures/sc_{opt}_results.pdf', bbox_inches = 'tight')
    print("Finished main figure!")


def gab_baseline_fig(ordered_datasets, dataset2plot_data, opt):
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
    if opt == 'ERM':
        fig, axs = plt.subplots(1, len(ordered_datasets), figsize=(17.5,3.5), sharey = True)
        plt.setp(axs, xlim=(0,1), ylim=(0, 1.01))
    else:
        fig, axs = plt.subplots(1, len(ordered_datasets), figsize=(17.5,3), sharey = True)
        plt.setp(axs, xlim=(0,1), ylim=(0.6, 1.005))
    for i, dataset in enumerate(ordered_datasets):
        plot_data = dataset2plot_data[dataset]
        worst_group = plot_data['worst_group']
        accuracies = plot_data['selective_accuracy']
        abstain_rates = plot_data['abstain_rate']
        coverages = plot_data['coverages']
        axs[i].plot(coverages[np.where(accuracies[worst_group] != -1)[0]], accuracies[worst_group][np.where(accuracies[worst_group] != -1)[0]], color = 'firebrick',
                   linewidth = 1.5)
        axs[i].plot(coverages, plot_data['gab_selective_accuracy'][worst_group], linestyle = 'dotted', color = 'firebrick', linewidth = 2)
        if opt == 'ERM':
            axs[i].plot(coverages, plot_data['robinhood_selective_accuracy'][worst_group], linestyle = 'dashed', color = 'firebrick', linewidth = 2)
        axs[i].plot(coverages[np.where(accuracies[-1] != -1)[0]], accuracies[-1][np.where(accuracies[-1] != -1)[0]], color = 'darkblue', linewidth = 2)
        if opt == 'ERM':
            legend = ['Worst group', 'Group-agnostic', 'Robin Hood', 'Average']
        else:
            legend = ['Worst group', 'Group-agnostic', 'Average']
        if opt == 'DRO' and dataset in ['CelebA']:
            axs[i].legend(legend, loc = 'lower left', fontsize = 14)
        elif opt == 'ERM' and dataset in ['MultiNLI']:
            axs[i].legend(legend, loc = 'lower right', fontsize = 14)
        axs[i].set_xticks(np.array([0,0.25, 0.5, 0.75, 1]))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axs[i].grid(b = True, linestyle = '--', color = '0.8')
        axs[i].set_title(dataset, fontsize = 20)
        ylabel = 'Selective accuracy'
        fontsize = 20 if opt == 'DRO' else 20
        axs[0].set_ylabel(ylabel, fontsize = fontsize)
        axs[i].set_xlabel('Coverage', fontsize = fontsize)
        axs[i].tick_params(axis='both', labelsize=16)
    fig.subplots_adjust(wspace=0.25, hspace=0)
    fig.savefig(f'figures/baseline_{opt}.pdf', bbox_inches='tight')

def mc_dropout_fig(ordered_datasets, dataset2plot_data):
    ordered_keys = ['accuracy', 'abstain_rate']
    plt.clf()
    group_dict = {i:i for i in range(-1,6)}
    datasets = list(dataset2plot_data)
    fig, axs = plt.subplots(2, len(ordered_datasets), figsize=(15, 6), sharey = True, sharex = True)

    for i, dataset in enumerate(ordered_datasets):
        plot_data = dataset2plot_data[dataset]
        n_groups = plot_data['n_groups']
        accuracies = plot_data['mc_selective_accuracy']
        abstain_rates = plot_data['mc_abstain_rate']
        coverages = plot_data['mc_coverages']
        axs[0,i].set_title(dataset, fontsize = 17)
        worst_group = plot_data['worst_group']
        groups = [n_groups - 1, -1] + [i for i in range(n_groups - 1)] + [-1]
        for group in groups:
            group_y = np.array(accuracies[group])
            rel_indices = np.where(group_y != -1)[0]
            color = '0.75'
            if group == worst_group:
                color = 'firebrick'
            elif group == -1:
                color = 'darkblue'
            axs[0,i].plot(plot_data['coverages'][rel_indices], group_y[rel_indices], color = color)
            axs[0, i].tick_params(axis='both', labelsize=16)
            axs[0, i].set_xticks(np.array([0,0.25, 0.5, 0.75, 1]))
            axs[0, i].set_yticks(np.array([0,0.25, 0.5, 0.75, 1]))
            axs[0, i].grid(b = True, linestyle = '--', color = '0.8')
        if dataset == 'MultiNLI':
            axs[0,i].legend(['Worst group', 'Average', 'Other groups'], fontsize = 13, loc = 'lower right')
        
        for group in groups:
            group_y = np.array(abstain_rates[group])
            rel_indices = np.where(group_y != -1)[0]
            color = '0.75'
            if group == worst_group:
                color = 'firebrick'
            elif group == -1:
                color = 'darkblue'
            axs[1,i].plot(plot_data['coverages'][rel_indices], 1 - group_y[rel_indices], color = color)
            axs[1, i].tick_params(axis='both', labelsize=16, length = 2)
            
            axs[1, i].set_xticks(np.array([0, 0.25, 0.5, 0.75, 1]))
            axs[1, i].set_xticklabels([0, None, 0.5, None, 1])
            axs[1, i].set_yticks(np.array([0, 0.25, 0.5, 0.75, 1]))
            axs[1, i].yaxis.set_major_formatter(FormatStrFormatter('%g'))
            axs[1, i].grid(b = True, linestyle = '--', color = '0.8')

    axs[0,0].set_ylabel('Group sel. accuracy', fontsize = 17)
    axs[1,0].set_ylabel('Group coverage', fontsize = 17)
    for i in range(len(datasets)):
        axs[1,i].set_xlabel('Average coverage', fontsize = 17)
    fig.subplots_adjust(wspace=0.25, hspace=0.1)
    fig.savefig(f'figures/mc_dropout_results.pdf', bbox_inches = 'tight')
    print("Done!")

def margin_plot(ordered_datasets, dataset2plot_data, opt):
    def softmax(x):
        x = x.astype(np.longdouble)
        assert len(x.shape) == 2
        x = x - x.max(axis = 1)[:,np.newaxis]
        x = np.exp(x) / np.exp(x).sum(axis = 1)[:, np.newaxis]
        return x

    def logit(x, n_classes = 2):
        return np.log(x/(1 - x))/2 + np.log(n_classes - 1)/2
        
    fig, axs = plt.subplots(2, len(ordered_datasets), figsize=(15, 3.5))

    for i, dataset in enumerate(ordered_datasets):
        plot_data = dataset2plot_data[dataset]
        labels = plot_data['labels']
        groups = plot_data['groups']
        scores = plot_data['scores']
        bin_preds = scores.argmax(axis = 1)
        n_classes = labels.max() + 1
        probs = softmax(scores)
        probs = probs[np.arange(probs.shape[0]), bin_preds]
        scores = logit(probs, n_classes = n_classes)
        #negate margin for incorrect predictions
        scores[np.where(bin_preds != labels)] = -1 * scores[np.where(bin_preds != labels)]
        max_score = np.abs(scores).max()
        legend = []
        axs[0, i].set_title(f'{dataset}', fontsize = 17)
        axs[0, 0].set_ylabel('Average\ndensity', fontsize = 17)
        axs[1, 0].set_ylabel('Worst-group\ndensity', fontsize = 17)
        axs[1, i].set_xlabel('Margin', fontsize = 16)

        axs[0, i].hist(scores, color = 'darkblue', bins = 50, weights = plot_data['shift_weights'], density = True, edgecolor = 'none')
        axs[0, i].set_xlim(-max_score, max_score)
        axs[0, i].tick_params(axis='both', labelsize=14)
        axs[0, i].set_yticks([])

        worst_group = plot_data['worst_group']
        group_scores = scores[np.where(groups == worst_group)[0]]
        axs[1, i].hist(group_scores, color = 'firebrick', bins = 50, density = True, edgecolor = 'none')
        axs[1, i].set_xlim(-max_score, max_score)
        axs[1, i].tick_params(axis='both', labelsize=14)
        axs[1, i].set_yticks([])
        axs[0, i].tick_params(labelbottom=False)    

    fig.align_ylabels(axs)
    fig.subplots_adjust(wspace=0.25, hspace=0.12)
    fig.savefig(f'figures/{opt}_margins.pdf', bbox_inches = 'tight')
    print("Done!")

if __name__ == '__main__':
    main()
