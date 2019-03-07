# Voice Conversion


## Overview of code: TODO: Update
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

# Dependencies
* PyTorch v1.0.0 or later
* numpy

# For preprocessing of VCTK dataset
* [torchaudio](https://github.com/pytorch/audio)
    * [libsox v14.3.2 or later](https://anaconda.org/conda-forge/sox)
    * GCC v4.9 or later
    * Remove the following silent samples before running preprocessing, `p323_424`, `p306_151`, `p351_361`, `p345_292`, `p341_101` 
