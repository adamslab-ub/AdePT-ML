import torch
import torch.nn
from adeptml import configs, models


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

    def __init__(self, config: configs.HybridConfig):
        super(HybridModel, self).__init__()
        self.models_nn = torch.nn.ModuleDict()
        self.models_physics = {}
        self.config = config
        for model_name in config.models:
            if isinstance(config.models[model_name], torch.nn.Module):
                self.models_nn[model_name] = config.models[model_name]().to(
                    configs.DEVICE
                )
            if isinstance(config.models[model_name], configs.PhysicsConfig):
                self.models_physics[model_name] = models.Physics.apply
            elif isinstance(config.models[model_name], configs.MLPConfig):
                self.models_nn[model_name] = models.MLP(config.models[model_name]).to(
                    configs.DEVICE
                )
            self.model_inputs = {}
            self.interim_data = {}
            if config.model_inputs:
                to_save = []
                for _, vals in config.model_inputs.items():
                    to_save += list(vals.keys())
                self.to_save = list(set(to_save))

    def forward(self, x, phy_args=None):
        """Function to run inference on the hybrid model."""
        self.interim_data["Input"] = x
        current_input = x
        for model_name in self.config.models:
            if self.config.model_inputs:
                if model_name in self.config.model_inputs:
                    input_tensors = []
                    for input_model, dims in self.config.model_inputs[
                        model_name
                    ].items():
                        if dims:
                            input_tensors.append(
                                self.interim_data[input_model][:, dims]
                            )
                        else:
                            input_tensors.append(self.interim_data[input_model])
                    current_input = torch.hstack(input_tensors)
            if model_name in self.models_nn.keys():
                cur_model = self.models_nn[model_name]
                out = cur_model(current_input)
            elif model_name in self.models_physics.keys():
                cur_model = self.models_physics[model_name]
                out = cur_model(
                    current_input,
                    self.config.models[model_name].forward_func,
                    self.config.models[model_name].jacobian_func,
                    phy_args,
                )
            if self.config.model_inputs:
                if model_name in self.to_save:
                    self.interim_data[model_name] = out
            current_input = out

        return out
