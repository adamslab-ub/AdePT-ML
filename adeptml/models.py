import torch
import torch.nn
from adeptml import configs

ACTIVATIONS = {
    "leakyrelu": torch.nn.LeakyReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
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
    def forward(ctx, x, forward_fun, jacobian_fun, args=None):
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
        input = ctx.saved_tensors[0]
        args = ctx.saved_tensors[1:]
        jacobian_fun = ctx.jacobian_fun
        jac_final = None
        if ctx.needs_input_grad[0]:
            input = input.detach().cpu().numpy()
            if args != None:
                args = [tmp_args.detach().cpu().numpy() for tmp_args in args]
                jac_final = jacobian_fun(input, *args)
                jac_final = torch.Tensor(jac_final).to(configs.DEVICE)
                jac_final = jac_final.reshape(input.shape[0], -1, input.shape[1])
                grad_output = grad_output.reshape(input.shape[0], -1, 1)
                grad_final = torch.matmul(grad_output, jac_final)
                return grad_final, None, None, None
            else:
                jac_final = jacobian_fun(input)
                jac_final = torch.Tensor(jac_final).to(configs.DEVICE)
                grad_final = torch.zeros(input.shape[0], input.shape[1]).to(
                    configs.DEVICE
                )
                grad_output = grad_output.reshape(input.shape[0], -1)
                for i in range(grad_final.shape[0]):
                    grad_final[i, :] = torch.matmul(
                        grad_output[i, :].reshape(1, -1),
                        jac_final[i].reshape(-1, input.shape[1]),
                    )
                return grad_final, None, None
