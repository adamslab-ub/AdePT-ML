import torch
import torch.nn
import numpy as np
from joblib import dump
import os
from hypyml import configs

ACTIVATIONS = {
    "leakyrelu": torch.nn.LeakyReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
}


class MLP(torch.nn.Module):
    """
    Multilayer Perceptron (MLP) neural network model.

    :param dict config: Configuration dictionary containing model parameters.
        - "Hidden_layer_size" (int): Size of the hidden layers.
        - "D_in" (int): Number of input dimensions.
        - "Num_layers" (int): Number of hidden layers.
        - "D_out" (int): Number of output dimensions.

    :ivar torch.nn.ModuleList layers: List of hidden layers in the MLP.
    :ivar torch.nn.BatchNorm1d norm: Batch normalization layer.
    :ivar torch.nn.Linear linear_in: Input linear layer.
    :ivar torch.nn.Linear linear_out: Output linear layer.
    :ivar torch.nn.LeakyReLU nl1: Leaky ReLU activation function.
    :ivar torch.nn.Tanh nl2: Hyperbolic tangent activation function.
    :ivar torch.nn.Sigmoid nl3: Sigmoid activation function.

    .. note::
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
    @staticmethod
    def forward(ctx, input, forward_fun, jacobian_fun, args=None):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        # ctx.needs_input_grad = (True,False)
        ctx.save_for_backward(input, args)
        ctx.jacobian_fun = jacobian_fun
        if args != None:
            out = forward_fun(input, args)
            return out
        else:
            out = forward_fun(input)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        (
            input,
            args,
        ) = ctx.saved_tensors
        jacobian_fun = ctx.jacobian_fun
        jac_final = None
        if ctx.needs_input_grad[0]:
            input = input.detach().cpu().numpy()
            if args != None:
                args = args.detach().cpu().numpy()
                jac_final = jacobian_fun(input, args)
                jac_final = torch.Tensor(jac_final).to(device)
                grad_final = torch.zeros(input.shape[0], input.shape[1]).to(device)
                grad_output = grad_output.reshape(input.shape[0], -1)
                for i in range(grad_final.shape[0]):
                    grad_final[i, :] = torch.matmul(
                        grad_output[i, :].reshape(1, -1),
                        jac_final[i].reshape(-1, input.shape[1]),
                    )
                return grad_final, None, None, None
            else:
                jac_final = jacobian_fun(input)
                jac_final = torch.Tensor(jac_final).to(device)
                grad_final = torch.zeros(input.shape[0], input.shape[1]).to(device)
                grad_output = grad_output.reshape(input.shape[0], -1)
                for i in range(grad_final.shape[0]):
                    grad_final[i, :] = torch.matmul(
                        grad_output[i, :].reshape(1, -1),
                        jac_final[i].reshape(-1, input.shape[1]),
                    )
                return grad_final, None, None
