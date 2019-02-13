# MLP Cluster Tutorial Branch
A short code repo that showcases a potential framework for carrying out experiments on the MLP Cluster.

## Introduction

Welcome to the MLPractical's Introduction to the MLP GPU Cluster branch. This branch provides tutorial material for the MLP Cluster.
The material available includes tutorial documents and code, as well as tooling that provides more advanced features to aid you in your quests
to train lots of learnable differentiable computational graphs.

## Getting Started
Before proceeding to the next section of the README, please read the [getting started guide](mlp_cluster_quick_start_up.md).

## Installation

The code uses Pytorch to run, along with many other smaller packages. To take care of everything at once, we recommend 
using the conda package management library. More specifically, 
[miniconda3](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), as it is lightweight and fast to install.
If you have an existing miniconda3 installation please start at step 3. 
If you want to  install both conda and the required packages, please run:
 1. ```wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh```
 2. Go through the installation.
 3. Activate conda
 4. conda create -n mlp python=3.6.
 5. conda activate mlp
 6. At this stage you need to choose which version of pytorch you need by visiting [here](https://pytorch.org/get-started/locally/)
 7. Choose and install the pytorch variant of your choice using the conda commands.
 8. Then run ```bash install.sh```

To execute an installation script simply run:
```bash <installation_file_name>```

To activate your conda installations simply run:
```conda activate```

## Overview of code:
- [arg_extractor.py](arg_extractor.py): Contains an array of utility methods that can parse python arguments or convert
 a json config file into an argument NamedTuple.
- [data_providers.py](data_providers.py): A sample data provider, of the same type used in the MLPractical course.
- [experiment_builder.py](experiment_builder.py): Builds and executes a simple image classification experiment, keeping track
of relevant statistics, taking care of storing and re-loading pytorch models, as well as choosing the best validation-performing model to evaluate the test set on.
- [model_architectures.py](model_architectures.py): Provides a fully connected network and convolutional neural network 
sample models, which have a number of moving parts indicated as hyperparameters.
- [storage_utils.py](storage_utils.py): Provides a number of storage/loading methods for the experiment statistics.
- [train_evaluated_emnist_classification_system.py](train_evaluate_emnist_classification_system.py): Runs an experiment 
given a data provider, an experiment builder instance and a model architecture

# Running an experiment
To run a default image classification experiment using the template models I provided:
1. Sign into the cluster using ssh sxxxxxxx@mlp1.inf.ed.ac.uk
2. Activate your conda environment using, source miniconda3/bin/activate ; conda activate mlp
3. cd mlpractical
4. cd cluster_experiment_scripts
5. Find which experiment(s) you want to run (make sure the experiment ends in 'gpu_cluster.sh'). Decide if you want to run a single experiment or multiple experiments in parallel.
    1. For a single experiment: ```sbatch experiment_script.sh```
    2. To run multiple experiments using the "hurdle-reducing" script that automatically submits jobs, makes sure the jobs are always in queue/running:
        1. Make sure the cluster_experiment_scripts folder contains ***only*** the jobs you want to run. 
        2. Run the command: 
        ```
        python run_jobs.py --num_parallel_jobs <number of jobs to keep in the slurm queue at all times> --num_epochs <number of epochs to run each job>
        ```
        