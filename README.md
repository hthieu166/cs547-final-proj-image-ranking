# CS547-Final Project Image Rannking

## Installation
### Setup
A conda environment file info `environment.yml` is given, to install all requires packages:
```
$ conda env -f environment.yml
$ conda activate py3torch
```
### Dataset
Download and extract the Tiny-ImageNet dataset from this link:
https://drive.google.com/file/d/1ATb_Xy-LsuT67ZmngSYaBabVvuEkBDSH/view?usp=sharing

Before starting any experiment, training and model configurations should be defined in
```
configs/model_cfgs
configs/train_cfgs
```
If you want to use your own model, loss, sampling objects, data augmentation strategy, etc. You can implement them inside the corrresponding folders inside `src/`. Remember to follow the interfaces and registered them under:
```
src/factories.py
```

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