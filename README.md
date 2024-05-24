# ADAMS Lab: Hybrid Python Machine Learning Library (HyPyML) Documentation:

This is a convienience library built on top of PyTorch to enable easy integration and training of hybrid models involving physics and deep learning modules. 

### Features
1. Pre-defined Modules and configs for physics and MLP architectures.
2. Physics Module accepts numpy arrays as outputs.
3. Ability to input custom PyTorch nn classes
4. Ensemble Module allows easy integration of constituent physics and ML modules for simple inference and training.
5. Auto log training data with tensorboard.

## Installation
Installing with pip
```
pip install hypyml
```
## Requirements (Automatically installed if the above installation command is used): 
1. PyTorch (https://pytorch.org/)
2. Joblib (https://joblib.readthedocs.io/en/latest/) (For loading and saving model parameters)

## Examples:
Refer to the tests file. Addtional examples will be added soon.

