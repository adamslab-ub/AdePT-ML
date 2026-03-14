import torch
import torch.nn
from typing import Callable, Optional, List
from adeptml import configs

ACTIVATIONS = {
    "identity": torch.nn.Identity(),
    "leakyrelu": torch.nn.LeakyReLU(),
    "relu": torch.nn.ReLU(),
    "elu": torch.nn.ELU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "sin": torch.sin,
    "softplus": torch.nn.Softplus(),
    "swish": torch.nn.SiLU(),
}


def _to_numpy(tensors):
    """Convert a list of tensors to numpy arrays. Returns an empty list for None/empty input."""
    if not tensors:
        return []
    return [t.detach().cpu().numpy() for t in tensors]


class MLP(torch.nn.Module):
    """
    Multilayer Perceptron (MLP) neural network model.

    Attributes
    ----------
    config : MLPConfig
        Instance of :class:`~adeptml.configs.MLPConfig`.

    Note
    ----
    This class implements a Multilayer Perceptron (MLP) neural network model.
    It takes a configuration dataclass with parameters such as hidden layer
    size, input and output dimensions, number of hidden layers, and activation
    functions.
    """

    def __init__(self, config: configs.MLPConfig):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.linear_in = torch.nn.Linear(config.num_input_dim, config.num_hidden_dim)
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                torch.nn.Linear(config.num_hidden_dim, config.num_hidden_dim)
            )
        self.linear_out = torch.nn.Linear(config.num_hidden_dim, config.num_output_dim)
        self.nl1 = ACTIVATIONS[config.hidden_activation]
        self.nl2 = ACTIVATIONS[config.output_activation]

    def forward(self, x):
        """
        Forward pass of the MLP model.

        :param torch.Tensor x: Input tensor.
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        out = self.linear_in(x)
        for i in range(len(self.layers)):
            net = self.layers[i]
            out = self.nl1(net(out))
        out = self.linear_out(out)
        return self.nl2(out)


class Physics(torch.autograd.Function):
    """Custom autograd function wrapping a physics model with a full Jacobian.

    The backward pass materialises the full ``(batch, out, in)`` Jacobian and
    contracts it with the upstream gradient via a batched matrix-vector product.
    Use :class:`Physics_VJP` when a manual VJP is cheaper than the full
    Jacobian, or :class:`Physics_SplitVJP` when the physics solver already
    produces a pullback closure (e.g. via ``jax.vjp``).

    See Also
    --------
    Physics_VJP, Physics_SplitVJP
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        forward_fun: Callable,
        jacobian_fun: Callable,
        args: Optional[List[torch.Tensor]] = None,
    ):
        """
        Run the physics forward pass.

        :param ctx: PyTorch autograd context.
        :param x: Input tensor ``(batch, in_dim)``.
        :param forward_fun: Physics function ``(x_np, *args_np) -> ndarray``.
        :param jacobian_fun: Jacobian function ``(x_np, *args_np) -> ndarray``
            with shape ``(batch, out_dim, in_dim)``.
        :param args: Extra positional arguments passed to ``forward_fun`` and
            ``jacobian_fun``.  Gradients are *not* computed w.r.t. these.
        :return: Output tensor ``(batch, out_dim)``.
        :rtype: torch.Tensor
        """
        if args:
            ctx.save_for_backward(x, *args)
        else:
            ctx.save_for_backward(x)
        ctx.jacobian_fun = jacobian_fun

        x_np = x.detach().cpu().numpy()
        args_np = _to_numpy(args)
        out = forward_fun(x_np, *args_np)
        return torch.tensor(out, dtype=x.dtype).to(configs.DEVICE)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute VJP via full Jacobian: ``grad = grad_output @ J``."""
        x = ctx.saved_tensors[0]
        args = ctx.saved_tensors[1:]
        jacobian_fun = ctx.jacobian_fun
        if ctx.needs_input_grad[0]:
            x_np = x.detach().cpu().numpy()
            args_np = _to_numpy(args)
            jac = jacobian_fun(x_np, *args_np)
            jac = torch.tensor(jac, dtype=grad_output.dtype).to(configs.DEVICE)
            jac = jac.reshape(x_np.shape[0], -1, x_np.shape[1])
            grad_final = torch.matmul(grad_output.unsqueeze(1), jac).squeeze(1)
            return grad_final, None, None, None
        return None, None, None, None


class Physics_VJP(torch.autograd.Function):
    """Custom autograd function wrapping a physics model with a manual VJP.

    Unlike :class:`Physics`, the backward pass delegates to a user-supplied
    ``jacobian_func(x, grad_output, *args) -> ndarray`` that returns the VJP
    directly, avoiding explicit Jacobian materialisation.

    See Also
    --------
    Physics, Physics_SplitVJP
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        forward_fun: Callable,
        jacobian_fun: Callable,
        args: Optional[List[torch.Tensor]] = None,
    ):
        """
        Run the physics forward pass.

        :param ctx: PyTorch autograd context.
        :param x: Input tensor ``(batch, in_dim)``.
        :param forward_fun: Physics function ``(x_np, *args_np) -> ndarray``.
        :param jacobian_fun: VJP function
            ``(x_np, grad_output_np, *args_np) -> ndarray``
            with shape ``(batch, in_dim)``.
        :param args: Extra positional arguments.  Gradients are *not* computed
            w.r.t. these.
        :return: Output tensor ``(batch, out_dim)``.
        :rtype: torch.Tensor
        """
        if args:
            ctx.save_for_backward(x, *args)
        else:
            ctx.save_for_backward(x)
        ctx.jacobian_fun = jacobian_fun

        x_np = x.detach().cpu().numpy()
        args_np = _to_numpy(args)
        out = forward_fun(x_np, *args_np)
        return torch.tensor(out, dtype=x.dtype).to(configs.DEVICE)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute VJP by calling the user-supplied VJP function directly."""
        x = ctx.saved_tensors[0]
        args = ctx.saved_tensors[1:]
        jacobian_fun = ctx.jacobian_fun
        if ctx.needs_input_grad[0]:
            x_np = x.detach().cpu().numpy()
            args_np = _to_numpy(args)
            grad_np = grad_output.detach().cpu().numpy()
            grad_final = jacobian_fun(x_np, grad_np, *args_np)
            grad_final = torch.tensor(grad_final, dtype=grad_output.dtype).to(
                configs.DEVICE
            )
            return grad_final, None, None, None
        return None, None, None, None


class Physics_SplitVJP(torch.autograd.Function):
    """Custom autograd function for physics models that expose a pullback closure.

    This mode supports ``forward_func`` functions that return both the output
    *and* a pullback closure — the pattern produced by ``jax.vjp``:

    .. code-block:: python

        y, pullback_fn = forward_func(x, *args)
        # later, during backward:
        (dx, *_) = pullback_fn(grad_output)

    The pullback closure is stored in the autograd context during the forward
    pass and invoked during the backward pass.  No part of the physics forward
    is re-executed during backpropagation, making this the most
    memory-efficient mode when the solver is expensive and already caches its
    intermediates (e.g. via ``jax.checkpoint`` inside ``jax.vjp``).

    ``jacobian_func`` is not used in this mode and should be omitted from
    :class:`~adeptml.configs.PhysicsConfig`.

    Example
    -------
    Given the JAX physics module::

        from Funcs import physics
        # physics(x)-> (y_normalized, pullback_fn)

    Configure and use as::

        from adeptml.configs import PhysicsConfig, HybridConfig
        from adeptml.ensemble import HybridModel

        phy_cfg = PhysicsConfig(
            forward_func=physics,
            use_split_vjp=True,
        )
        hybrid_cfg = HybridConfig(models={"nn": mlp_cfg, "physics": phy_cfg})
        model = HybridModel(hybrid_cfg)

        # During training the pullback is cached automatically; no extra code
        # is needed in the training loop.

    See Also
    --------
    Physics, Physics_VJP
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        forward_fun: Callable,
        args: Optional[List[torch.Tensor]] = None,
    ):
        """
        Run the split-VJP physics forward pass.

        Calls ``forward_fun(x_np, *args_np)`` which must return
        ``(output_np, pullback_fn)``.  The pullback is stored in ``ctx`` for
        use during :meth:`backward`.

        :param ctx: PyTorch autograd context.
        :param x: Input tensor ``(batch, in_dim)``.
        :param forward_fun: Split-VJP physics function
            ``(x_np, *args_np) -> (ndarray, callable)``.
        :param args: Extra positional arguments (e.g. boundary conditions,
            initial conditions, simulation time).  Gradients are *not* computed
            w.r.t. these.
        :return: Output tensor ``(batch, out_dim)``.
        :rtype: torch.Tensor
        """
        if args:
            ctx.save_for_backward(x, *args)
        else:
            ctx.save_for_backward(x)

        x_np = x.detach().cpu().numpy()
        args_np = _to_numpy(args)
        out_np, pullback_fn = forward_fun(x_np, *args_np)
        ctx.pullback_fn = pullback_fn
        return torch.tensor(out_np, dtype=x.dtype).to(configs.DEVICE)

    @staticmethod
    def backward(ctx, grad_output):
        """Call the stored pullback closure with the upstream gradient.

        The cotangent for ``x`` is ``pullback_fn(grad_output)[0]``; cotangents
        for ``forward_fun`` and ``args`` are ``None``.
        """
        if ctx.needs_input_grad[0]:
            grad_np = grad_output.detach().cpu().numpy()
            cotangents = ctx.pullback_fn(grad_np)
            # cotangents is a tuple ordered by the arguments of forward_fun;
            # index 0 corresponds to x.
            dx = torch.tensor(cotangents[0], dtype=grad_output.dtype).to(
                configs.DEVICE
            )
            return dx, None, None
        return None, None, None
