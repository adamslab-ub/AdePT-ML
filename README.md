# Auto-differentiable embedding of Physics and Torch Machine Learning (AdePT-ML):

This is a convienience library built on top of PyTorch to enable easy integration and training of hybrid models involving physics and deep learning modules. 

### Features
1. Allows integration of torch.nn.module with numpy functions and enable training with torch optimizers.
1. Pre-defined Modules and configs for physics and MLP architectures.
5. Integrated training function with tensorboard support.

## Installation
Installing with pip
```
pip install adeptml 
```
### Requirements (Automatically installed with pip): 
1. PyTorch (https://pytorch.org/)
2. Joblib (https://joblib.readthedocs.io/en/latest/) (For loading and saving model parameters)
3. Tensorboard

## Usage:

The primary building block of this package is the [Hybrid Model]() class. It neatly packages all the member models into one main Torch model and enables running forward inference as well as backpropagation.
The class accepts as input an instance of the [Hybrid Config]() class. This config is useful in defining all the constituent modules and their inputs.

As component modules, the [Models]() module provides a straight forward [MLP]() implementation as well as a [Physics Module]().
This module is a torch Autograd wrapper which enables the integration of non-Torch numpy functions into a fully torch model and allows for training with torch optimizers.

## API Documentation:
Visit our [Read the docs page](https://adept-ml.readthedocs.io/en/latest/)

## Examples:
Refer to the tests file. Additional examples will be added soon.

