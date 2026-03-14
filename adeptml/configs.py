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

    hidden_activation : str
        Activation function for hidden layers.
        Choices are ``"identity"``, ``"relu"``, ``"leakyrelu"``, ``"elu"``,
        ``"sigmoid"``, ``"tanh"``, ``"sin"``, ``"softplus"``, ``"swish"``.

    output_activation : str
        Activation function for the output layer.
        Choices are ``"identity"``, ``"relu"``, ``"leakyrelu"``, ``"elu"``,
        ``"sigmoid"``, ``"tanh"``, ``"sin"``, ``"softplus"``, ``"swish"``.
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

    Three backpropagation modes are supported, selected via ``use_vjp`` and
    ``use_split_vjp``:

    **Jacobian mode** (default, ``use_vjp=False``, ``use_split_vjp=False``)
        ``forward_func(x, *args) -> ndarray``
        ``jacobian_func(x, *args) -> ndarray``  shape ``(batch, out, in)``

        ``backward`` forms the VJP by multiplying ``grad_output @ jacobian``.
        Use when the full Jacobian is cheap or already available.

    **VJP mode** (``use_vjp=True``, ``use_split_vjp=False``)
        ``forward_func(x, *args) -> ndarray``
        ``jacobian_func(x, grad_output, *args) -> ndarray``  shape ``(batch, in)``

        ``backward`` calls ``jacobian_func`` with the upstream gradient directly,
        avoiding explicit Jacobian materialisation.  Use when a manual VJP is
        cheaper than the full Jacobian.

    **Split-VJP mode** (``use_split_vjp=True``)
        ``forward_func(x, *args) -> (ndarray, pullback_fn)``

        ``forward_func`` returns both the output *and* a pullback closure (e.g.
        the result of ``jax.vjp``).  The closure is stored during the forward
        pass and called with ``grad_output`` during the backward pass without
        re-running any part of the physics.  ``jacobian_func`` is not used and
        may be omitted.

        This is the most memory-efficient mode when the physics solver is
        expensive and intermediates are already cached by the underlying
        framework (e.g. JAX checkpointing via ``jax.checkpoint``).

        Example usage with a JAX physics function::

            from my_physics import physics_forward
            # physics_forward(x, *args) -> (y, pullback_fn)

            phy_cfg = PhysicsConfig(
                forward_func=physics_forward,
                use_split_vjp=True,
            )

    Attributes
    ----------
    forward_func : Callable
        Physics forward function.  Signature depends on the chosen mode — see
        above.

    jacobian_func : Callable, optional
        Jacobian or VJP function.  Required for Jacobian and VJP modes; not
        used in split-VJP mode.

    use_vjp : bool
        Enable VJP mode.  Ignored when ``use_split_vjp=True``.

    use_split_vjp : bool
        Enable split-VJP mode.  Takes precedence over ``use_vjp``.
    """

    forward_func: Callable
    jacobian_func: Optional[Callable] = None
    use_vjp: bool = False
    use_split_vjp: bool = False


ModelConfig = Union[MLPConfig, PhysicsConfig]


@dataclasses.dataclass
class HybridConfig:
    """
    Configuration for Hybrid (ensemble) models.

    Attributes
    ----------
    models : dict
        Maps model names (str) to a :class:`ModelConfig` or an existing
        ``torch.nn.Module``.  Models are executed in insertion order.

    model_inputs : dict, optional
        By default the ensemble runs sequentially — the output of each model
        becomes the input of the next.  Setting this dict overrides that
        behaviour for specific models.

        Keys are model names; values are dicts mapping *source model names* to
        dimension selectors:

        - ``None`` — pass the entire source tensor.
        - ``[int, ...]`` — pass only the listed dimensions (``tensor[:, dims]``).

        Use the special key ``"Input"`` to reference the original input to
        :meth:`HybridModel.forward`.

        Example — concatenate dims 0-3 of the original input with the full
        output of ``"mlp1"``::

            model_inputs={
                "physics": {
                    "Input": [0, 1, 2, 3],
                    "mlp1": None,
                }
            }
    """

    models: dict[str, ModelConfig | torch.nn.Module]
    model_inputs: Optional[dict[str, dict[str, list[int] | None]]] = None
