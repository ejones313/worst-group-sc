# Selective Classification Can Magnify Disparities Across Groups
This repository contains the code used for the following paper:
> [**Selective Classification Can Magnify Disparities Across Groups**](https://arxiv.org/abs/2010.14134)
> 
> Erik Jones\*, Shiori Sagawa\*, Pang Wei Koh\*, Ananya Kumar, and Percy Liang
> 
> International Conference on Learning Representations (ICLR), 2021
> 
For an executable version of our paper, check out the [CodaLab Worksheet](https://worksheets.codalab.org/worksheets/0x7ceb817d53b94b0c8294a7a22643bf5e).
## Abstract 
Selective classification, in which models can abstain on uncertain predictions, is a natural approach to improving accuracy in settings where errors are costly but abstentions are manageable. In this paper, we find that while selective classification can improve average accuracies, it can simultaneously magnify existing accuracy disparities between various groups within a population, especially in the presence of spurious correlations. We observe this behavior consistently across five vision and NLP datasets. Surprisingly, increasing abstentions can even decrease accuracies on some groups. To better understand this phenomenon, we study the margin distribution, which captures the model’s confidences over all predictions. For symmetric margin distributions, we prove that whether selective classification monotonically improves or worsens accuracy is fully determined by the accuracy at full coverage (i.e., without any abstentions) and whether the distribution satisfies a property we call left-log-concavity. Our analysis also shows that selective classification tends to magnify full-coverage accuracy disparities. Motivated by our analysis, we train distributionally-robust models that achieve similar full-coverage accuracies across groups and show that selective classification uniformly improves each group on these models. Altogether, our results suggest that selective classification should be used with care and underscore the importance of training models to perform equally well across groups at full coverage.

## Data
We use five datasets in our experiments, four of which are available in the correct format as downloadable bundles from CodaLab: [CelebA (Liu et al., 2015)](https://worksheets.codalab.org/bundles/0x886412315184400c9983b32846e91ab1), [Waterbirds (Sagawa et al., 2020)](https://worksheets.codalab.org/bundles/0xb922b6c2d39c48bab4516780e06d5649), [CivilComments (Borkan et al., 2019)](https://worksheets.codalab.org/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e), and [MultiNLI (Williams et al., 2018)](https://worksheets.codalab.org/bundles/0xeb40a0a0277840b7b1fd057008bd25f5).
We additionally use a modified version of [CheXpert (Irvin et al., 2019)](https://stanfordmlgroup.github.io/competitions/chexpert/), called CheXpert-device, where we subsample to enforce a spurious correlation between the presence of pleural effusion and the presence of a support device. 
The splits we use are available on [CodaLab](https://worksheets.codalab.org/rest/bundles/0x0ea792f3d6b74e65bbbe76086b0704ce/contents/blob/chexpert_paths.csv).

## Running
There are two main steps in reproducing the results of our paper:

1. Training models and saving predictions
2. Plotting results based on saved predictions

### Training models
The code for the first step is stored in `src`, and is heavily based off of [this code](https://github.com/kohpangwei/group_DRO). 
As an example, consider the following command:

```
python3 src/run_expt.py -d Waterbirds -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 128 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --data_dir waterbirds --log_dir ./preds --save_preds --save_step 1000 --log_every 1000
```

Here, replace the `--data_dir` argument with the location of the Waterbirds bundle downloaded from CodaLab. The `-d` argument specifies the dataset, the `-t` specifies the label, the `-c` specifies the name of the confounder, and the predictions on the test set will be stored in `--log_dir`. In this case, the optimizer is ERM; to change to DRO, add `--robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 1`, and to add dropout add `--mc_dropout`. Commands to train each model for each dataset are available on CodaLab, except for CheXpert. For CheXpert, request access from the bottom of [this page](https://stanfordmlgroup.github.io/competitions/chexpert/). Then, run

```
python3 src/run_expt.py -s confounder -d CheXpert -t Pleural_Effusion -c Support_Devices --lr 0.001 --batch_size 16 --weight_decay 0 --model densenet121 --n_epochs 4 --log_every 10000 --log_dir ./preds --data_dir CheXpert-v1.0-small
```

Here, `CheXpert-v1.0-smal`l is the folder containing the small version of the downloaded CheXpert dataset. You will first need to filter the downloaded `metadata.csv` file with to only contain entries with `Path` contained within the `chexpert_paths.csv` file available on codalab. [CodaLab](https://worksheets.codalab.org/rest/bundles/0x0ea792f3d6b74e65bbbe76086b0704ce/contents/blob/chexpert_paths.csv)
Ensure that the `split` column from `chexpert_paths.csv` also translates over. 

### Plotting resultsd
Next, given saved models, we compute and plot the accuracy-coverage curves, along with the group-agnostic reference, the Robin Hood reference, and the margin distributions. To do so, ensure the saved `preds` folder from the previous step, for each `dataset` for ERM, ERM with dropout, and MC-Dropout, are stored in `dataset-ERM`, `dataset-DRO`, and `dataset-MC` respectively. Then, to plot figures, run:

```
python3 eval/process_preds.py --opt ERM --datasets CelebA Waterbirds CheXpert-device CivilComments MultiNLI
```

Feel free to remove and reorder datasets depending on the desired figures, and replace `ERM` with `DRO` and `MC` to generate plots for models trained with DRO and MC-dropout based confidences respectively. The output is stored in `figures`. 
