import torch
import torch.nn
from typing import Callable, Optional, List
from adeptml import configs

ACTIVATIONS = {
    "leakyrelu": torch.nn.LeakyReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
}


class MLP(torch.nn.Module):
    """
    Multilayer Perceptron (MLP) neural network model.

    Attributes
    ----------
    config : Instance of MLPConfig dataclass.

    Note
    ----
    This class implements a Multilayer Perceptron (MLP) neural network model.
    It takes a configuration dictionary with parameters such as hidden layer size,
    input and output dimensions, and the number of hidden layers.
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
        self.nl1 = ACTIVATIONS[config.activation_functions]

    def forward(self, x):
        """
        Forward pass of the MLP model.

        :param torch.Tensor x: Input tensor.

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        out = self.linear_in(x)
        for i in range(len(self.layers) - 1):
            net = self.layers[i]
            out = self.nl1(net(out))
        out = self.linear_out(out)
        return out


class Physics(torch.autograd.Function):
    """Custom Autograd function to enable backpropagation on Custom Physics Models.

    Attributes:
    config: Instance of PhysicsConfig.
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
        Function defining forward pass for the physics model.

        :param ctx: Torch Autograd context object (https://pytorch.org/docs/stable/autograd.html#context-method-mixins)
        :param x: Input tensor
        :param forward_fun: Function which computes outputs of the physics function. Accepts numpy arrays as input.
        :param jacobian_fun: Function which computes Jacobian / gradient of the physics function. Accepts numpy arrays as input.
        :param args: List containing additional positional arguments (as tensors) to forward_fun. Gradients are not computed w.r.t these args.

        :return: The output of forward_fun as a tensor.
        """
        if args:
            ctx.save_for_backward(x, *args)
        else:
            ctx.save_for_backward(x)
        ctx.jacobian_fun = jacobian_fun
        x = x.detach().cpu().numpy()
        if args != None:
            args = [tmp_args.detach().cpu().numpy() for tmp_args in args]
            out = forward_fun(x, *args)
            out = torch.Tensor(out).to(configs.DEVICE)
            return out
        else:
            out = forward_fun(x)
            out = torch.Tensor(out).to(configs.DEVICE)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Function to compute gradient across the forward_fun during backpropagation.
        """
        input = ctx.saved_tensors[0]
        args = ctx.saved_tensors[1:]
        jacobian_fun = ctx.jacobian_fun
        jac_final = None
        if ctx.needs_input_grad[0]:
            input = input.detach().cpu().numpy()
            if args is not None:
                args = [tmp_args.detach().cpu().numpy() for tmp_args in args]
                jac_final = jacobian_fun(input, *args)
            else:
                jac_final = jacobian_fun(input)
            jac_final = torch.Tensor(jac_final).to(configs.DEVICE)
            jac_final = jac_final.reshape(input.shape[0], -1, input.shape[1])
            grad_output = grad_output.unsqueeze(1)
            grad_final = torch.matmul(grad_output, jac_final).squeeze()
            return grad_final, None, None, None
