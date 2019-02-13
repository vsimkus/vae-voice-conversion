# Pytorch Experiment Framework

## What does this framework do?
The Pytorch experiment framework located in ```mlp/pytorch_experiment_scripts``` includes tooling for building an array of deep neural networks,
including fully connected and convolutional networks. In addition, it also includes tooling for experiment running, 
metric handling and storage, model weight storage, checkpointing (allowing continuation from previous saved point), as 
well as taking care of keeping track of the best validation model which is then used as the end to produce test set evaluation metrics.

## Why do we need it?
It serves two main purposes. The first, is to allow you an easy, worry-free transition into using Pytorch for experiments
 in your coursework. The second, is to teach you good coding practices for building and running deep learning experiments
  using Pytorch. The framework comes fully loaded with tooling that can keep track of relevant metrics, save models, resume from previous saved states and 
  even automatically choose the best val model for test set evaluation. We include documentation and comments in almost 
  every single line of code in the framework, to help you maximize your learning. The code style itself, can be used for
   learning good programming practices in structuring your code in a modular, readable and computationally efficient manner that minimizes chances of user-error.

## Installation

First thing you have to do is activate your conda MLP environment. 

### GPU version on Google Compute Engine
For usage on google cloud, the disk image we provide comes pre-loaded with all the packages you need to run the Pytorch
experiment framework, including Pytorch itself.  Thus when you created an instance and setup your environment, everything you need for this framework was installed, thus removing the need for you to install Pytorch.



### CPU version on DICE (or other local machine)

If you do not have your MLP conda environment installed on your current machine 
please follow the instructions in notes/environment-set-up.md. Once your mlp conda environment is activated, please go to
[Pytorch's installation page](https://pytorch.org/get-started/locally/) and take some time to choose the right Pytorch version for your setup (taking care to choose CPU/GPU version depending on what hardward you have available).

For example, on DICE you can install the CPU version using the command: 
```
conda install pytorch-cpu torchvision-cpu -c pytorch
```

Once Pytorch is installed in your mlp conda enviroment, you can start using the framework. The framework has been built 
to allow you to control your experiment hyperparameters directly from the command line, by using command line argument parsing.

## Using the framework

You can get a list of all available hyperparameters and arguments by using:
```
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py -h
```

The -h at the end is short for --help, which presents a list with all possible arguments next to a description of what they modify in the setup.
Once you execute that command, you should be able to see the following list:

```
Welcome to the MLP course's Pytorch training and inference helper script

optional arguments:
  -h, --help            show this help message and exit
  --batch_size [BATCH_SIZE]
                        Batch_size for experiment
  --continue_from_epoch [CONTINUE_FROM_EPOCH]
                        Batch_size for experiment
  --seed [SEED]         Seed to use for random number generator for experiment
  --image_num_channels [IMAGE_NUM_CHANNELS]
                        The channel dimensionality of our image-data
  --image_height [IMAGE_HEIGHT]
                        Height of image data
  --image_width [IMAGE_WIDTH]
                        Width of image data
  --dim_reduction_type [DIM_REDUCTION_TYPE]
                        One of [strided_convolution, dilated_convolution,
                        max_pooling, avg_pooling]
  --num_layers [NUM_LAYERS]
                        Number of convolutional layers in the network
                        (excluding dimensionality reduction layers)
  --num_filters [NUM_FILTERS]
                        Number of convolutional filters per convolutional
                        layer in the network (excluding dimensionality
                        reduction layers)
  --num_epochs [NUM_EPOCHS]
                        The experiment's epoch budget
  --experiment_name [EXPERIMENT_NAME]
                        Experiment name - to be used for building the
                        experiment folder
  --use_gpu [USE_GPU]   A flag indicating whether we will use GPU acceleration
                        or not (defaults to CPU)
  --weight_decay_coefficient [WEIGHT_DECAY_COEFFICIENT]
                        Weight decay to use for Adam

```

For example, to run a simple experiment using a convolutional network and max-pooling on the CPU you can run:

```
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type max_pooling --experiment_name tutorial_exp_1 --use_gpu False
```

Your experiment should begin running.

Your experiments statistics and model weights are saved in the directory tutorial_exp_1/ under tutorial_exp_1/logs and 
tutorial_exp_1/saved_models.


To run on a GPU on Google Compute Engine the command would be:
```
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type max_pooling --experiment_name tutorial_exp_1 --use_gpu True
```



## So, where can I ask more questions and find more information on Pytorch and what it can do?

First course of action should be to search the web and then to refer to the Pytorch [documentation](https://pytorch.org/docs/stable/index.html),
 [tutorials](https://pytorch.org/tutorials/) and [github](https://github.com/pytorch/pytorch) sites.
 
 If you still can't get an answer to your question then as always, post on Piazza and/or come to the lab sessions.
 