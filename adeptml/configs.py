import dataclasses
from typing import Callable, Optional, Union
import torch


if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


@dataclasses.dataclass
class MLPConfig:
    """
    Configuration class for the Multilayer Perceptron (MLP) model.

    Attributes
    ----------
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
        Choices are "leakyrelu", "sigmoid" and "tanh"
    """

    num_input_dim: int
    num_hidden_dim: int
    num_output_dim: int
    num_hidden_layers: int
    hidden_activation: str
    output_activation: str


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


ModelConfig = Union[MLPConfig, PhysicsConfig]


@dataclasses.dataclass
class HybridConfig:
    """
    Config for Ensemble Models.

    Attributes
    ----------
    models: dict
        Contains Modelname as keys and an instance of ModelConfigs as values.

    model_inputs: dict
        By default, the Ensemble model operates sequentially, using the output of the preceding model as input for the next.
        Setting this dict to a non-empty value overrides that behavior.
        Keys are model names; values are dicts specifying input customization.
        Each inner dict holds model names as keys and specifies how to stack inputs:
        - 'None' stacks the entire tensor.
        - A list of ints stacks only specified dimensions.
        Use "Input" if the input to this model matches the hybrid model's original input.
    """

    models: dict[str, ModelConfig | torch.nn.Module]
    model_inputs: Optional[dict[str, dict[str, list[int] | None]]] = None
