# [CS547/SP21/GROUP 31 - Final Project] Image Ranking
GitHub Repo: https://github.com/hthieu166/cs547-final-proj-image-ranking
## Authors
Hieu Hoang; 	Dong Kai Wang;		Kesav Shivam;	Yashaswini Murthy

{hthieu, dwang47, kshivam2, ymurthy}@illinois.edu

## Installation
### Setup
A conda environment file info `environment.yml` is given, to install all requires packages:
```
$ conda env -f environment.yml
$ conda activate py3torch
```
### Dataset
* Download and extract Tiny-ImageNet 200 dataset from this link:
https://drive.google.com/file/d/1ATb_Xy-LsuT67ZmngSYaBabVvuEkBDSH/view?usp=sharing
* Download and extract Market-1501 dataset from this link:
https://www.kaggle.com/pengcw1/market-1501/data

### Setup

Before starting any experiment, training and model configurations should be defined in
```
configs/model_cfgs
configs/train_cfgs
```
If you want to use your own model, loss, sampling objects, data augmentation strategy, etc. You can implement them inside the corrresponding folders inside `src/`. Remember to follow the interfaces and registered them under:
```
src/factories.py
```

Implementation of models, online sampling techniques and loss functions are available uder `src/`

### Training
Open a script file (e.g. `scripts/image_ranking.sh`), change `--is_training` flag to `true` and execute the script:
```
$ ./scripts/image_ranking.sh ${GPU_ID}
```
For your convenience, the default `GPU_ID` and `N_WORKERS` can be assigned under `scripts/master_env.sh` 
Results are given in the `logs/` folder, they includes the model (best score and checkpoints), log files for Tensorboard, and stdout. To use Tensorboard, under your log directory:
```
$ tensorboard --logdir=./ --port [YOUR PORT NUMBER]
```

### Testing
Similarly, change `--is_training` flag to `false`, specify the directory output under `--output` flag and the weight of your model under `--pretrained_model_path`

## Logs Folder
Logs folder of our experiments is available at:
https://drive.google.com/drive/folders/18iSUuAwz_eeA9HM3gcaMO66DPUiIaCdI?usp=sharing