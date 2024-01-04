import dataclasses
from typing import NewType, Callable, Optional
import torch
import typeddict


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

    Note
    ----
    This class serves as a configuration class for the Multilayer Perceptron (MLP) model.
    It defines various parameters that can be used to customize the architecture of the MLP.
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

    arg_dimensions : tuple[int, ...]
        List of integer dimensions representing the arguments of the physics functions.

    Note
    ----
    This class serves as a configuration class for physics-related functions.
    It defines various parameters that can be used to configure the behavior of the physics model.
    """

    forward_func: Callable[[torch.Tensor], torch.Tensor]
    jacobian_func: Callable[[torch.Tensor], torch.Tensor]


ModelConfig = NewType("ModelConfig", [MLPConfig | PhysicsConfig])


@dataclasses.dataclass
class EnsembleConfig:
    """Config for Ensemble Models."""

    models: dict[str, ModelConfig]
    non_serial_inputs: Optional[dict[str, str]] = None
