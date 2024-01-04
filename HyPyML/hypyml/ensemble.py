from platform import architecture
import torch
import torch.nn
from hypyml import models, configs
from immutabledict import immutabledict


class HybridModel(torch.nn.Module):
    """
    Torch Module for Serial Hybrid Physics Models.

    Parameters
    ----------
    models_mlp : Torch module list of all MLP modules.

    models_cnn : Torch module list of all CNN modules.

    models_physics : Torch module list of all Physics modules.

    unmodified_inputs : Indices of the inputs that are to be passed directly to the
    model. These are appended to the outputs of the previous model.

    architecture : Dict with key corresponding to model name and value being a model
    config.
    """

    def __init__(self, config: dict[str, configs.ModelConfig]):
        super(HybridModel, self).__init__()
        self.models_nn = torch.nn.ModuleDict()
        self.models_physics = {}
        self.unmodified_inputs = []
        self.config = config
        self.interm_inputs = {}
        self.interm_outputs = {}
        # self.model_ids =
        for model_name in config:
            if isinstance(config[model_name], configs.PhysicsConfig):
                self.models_physics[model_name] = models.Physics.apply
            elif isinstance(config[model_name], configs.MLPConfig):
                self.models_mlp[model_name] = models.MLP(config[model_name])
            if model_name in config.non_serial_inputs.keys():
                assert (
                    config.non_serial_inputs[model_name] in config.models.keys()
                    or self.config.non_serial_inputs[model_name] == "Input"
                ), f"No model named {config.non_serial_inputs[model_name]} in models dict"
                self.interm_inputs[config.non_serial_inputs[model_name]] = None
                self.interm_outputs[model_name] = None

    def forward(self, x):
        """
        Helper function to run inference on one constituent model.
        """
        current_input = x
        for model_name in self.config.models:
            if model_name in self.config.non_serial_inputs.keys():
                if self.config.non_serial_inputs[model_name] != "Input":
                    current_input = self.interm_inputs[
                        self.config.non_serial_inputs[model_name]
                    ]
                else:
                    current_input = x
            if model_name in self.models_nn.keys():
                cur_model = self.models_nn[model_name]
                if self.config.models[model_name].args != None:
                    out = cur_model(
                        torch.hstack(
                            (x[:, self.config.models[model_name].args], current_input)
                        )
                    )
                else:
                    out = cur_model(current_input)
            elif model_name in self.models_physics.keys():
                cur_model = self.models_physics[model_name]
                if self.config[model_name].arg_dimensions != None:
                    out = cur_model(
                        current_input,
                        self.config[model_name].forward_func,
                        self.config[model_name].jacobian_func,
                        x[:, self.config.models[model_name].args],
                    )
                else:
                    out = cur_model(
                        current_input,
                        self.config.models[model_name].forward_func,
                        self.config.models[model_name].jacobian_func,
                    )
            if model_name in self.config.non_serial_inputs.keys():
                self.interm_outputs[model_name] = out

        if self.config.non_serial_inputs:
            output = 0
            for i in self.interm_outputs:
                output += self.interm_outputs[i]

        return output
