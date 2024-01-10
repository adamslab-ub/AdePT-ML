import dataclasses
from typing import NewType, Callable, Optional
import torch
import typeddict

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


@dataclasses.dataclass
class MLPConfig:
    """
    Configuration class for the Multilayer Perceptron (MLP) model.

    Attributes
    ----------
    layers : int
        Total number of layers in the MLP (including hidden and output layers).

    num_input_dim : int
        Number of input dimensions to the MLP.

    num_hidden_dim : int
        Number of hidden dimensions in each hidden layer.

    num_output_dim : int
        Number of output dimensions from the MLP.

    num_hidden_layers : int
        Number of hidden layers in the MLP.

    activation_functions : str
        String representation of the activation functions used in the MLP.
    """

    num_input_dim: int
    num_hidden_dim: int
    num_output_dim: int
    num_hidden_layers: int
    activation_functions: str


@dataclasses.dataclass
class PhysicsConfig:
    """
    Configuration class for physics-related functions.

    Attributes
    ----------
    forward_func : Callable[[torch.Tensor], torch.Tensor]
        Forward function of the physics model.

    jacobian_func : Callable[[torch.Tensor], torch.Tensor]
        Function to compute the Jacobian of the physics model.
    """

    forward_func: Callable[[torch.Tensor], torch.Tensor]
    jacobian_func: Callable[[torch.Tensor], torch.Tensor]


ModelConfig = NewType("ModelConfig", [MLPConfig | PhysicsConfig])


@dataclasses.dataclass
class HybridConfig:
    """
    Config for Ensemble Models.

    Attributes
    ----------
    models : dict
        Dictionary containing Modelname as the key and an instance of ModelConfigs as the value.
    io_overrides : dict
        By default, the Ensemble model operates in serial mode, where the output of the preceding model
        is used as the input for the current model.
        Setting this dictionary to a non-empty value will override that behavior.
        If a model doesn't take the output of the preceding model as input or requires additional inputs,
        the names of those models should be the keys in this dictionary.
        The corresponding values should be tuples containing the names of models whose output needs to be provided to this model.
        Use "Input" as the value if the input to this model is the same as the original input to the hybrid model.

    """

    models: dict[str, ModelConfig]
    io_overrides: Optional[dict[str, tuple[str, ...]]] = None
