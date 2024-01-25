# ADAMS Lab: Hybrid Python Machine Learning Library (HyPyML) Documentation:
The following document describes usage of the HyPyML python library.

<!-- ## To Do:
### Features
- [ ] Add ability to input custom PyTorch nn classes
- [ ] For each module, optionally specify the order of input and output. Default is just serial
### Documentation
- [ ] Add documentation to every class allowing for the future use of sphinx.
- [ ] Update the sample script and example documentation -->

## Installation


Installing with the built wheel (Pre-built wheel available for Ubuntu 22.04 based distros)
```
pip install dist/hypyml-x.x.x-py3-none-any.whl
```
Building wheel using [poetry](https://python-poetry.org/).
 
When installing on a OS other than the one specified above, the wheel can be built using the following command after which the same installation command is to be used.

```
cd HyPyML
poetry build
```

## Requirements (Automatically installed if the above installation command is used): 
1. Google JAX (https://github.com/google/jax)
2. PyTorch (https://pytorch.org/)
3. Joblib (https://joblib.readthedocs.io/en/latest/)

## Package Classes:
This package offers 2 modules (MLP and Physics) that can be used as part of the hybrid model. 
The MLP class is a torch.nn module that is initialized by an instance of MLPConfig, while the Physics module is a custom torch autograd function.

The hybrid model class needs an instance of configs.HybridConfig to be initialized and each of the constituent modules also need an instance of their configs.

## Training 
Please refer to the **tests.py** file provided.