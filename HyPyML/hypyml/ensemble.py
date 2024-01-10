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

    def __init__(self, config: configs.HybridConfig):
        super(HybridModel, self).__init__()
        self.models_nn = torch.nn.ModuleDict()
        self.models_physics = {}
        self.config = config
        # self.model_ids =
        for model_name in config.models:
            if isinstance(config.models[model_name], configs.PhysicsConfig):
                self.models_physics[model_name] = models.Physics.apply
            elif isinstance(config.models[model_name], configs.MLPConfig):
                self.models_nn[model_name] = models.MLP(config.models[model_name])
        io_map = {
            v: k
            for (v, k) in zip(
                config.models.keys(), ["Input"] + list(config.models.keys())[:-1]
            )
        }
        if config.io_overrides:
            to_save = []
            for entry in config.io_overrides.keys():
                io_map[entry] = config.io_overrides[entry]
                to_save.append(entry)
                to_save += config.io_overrides[entry]
            to_save = list(set(to_save))
            self.interim_data = {i: None for i in to_save}
            used_inputs = []
            for i in io_map.values():
                if isinstance(i, tuple):
                    used_inputs += list(i)
                elif isinstance(i, str):
                    used_inputs.append(i)
                else:
                    raise TypeError(
                        f"Values in io_overrides need to be tuple or str not {type(i)}"
                    )
            print(self.interim_data.keys())
            used_inputs = set(used_inputs)
            self.outputs = list(set(config.models.keys()) - used_inputs)

    def forward(self, x):
        """Function to run inference on the hybrid model."""
        current_input = x
        for count, model_name in enumerate(self.config.models):
            if self.config.io_overrides and count > 0:
                if model_name in self.config.io_overrides.keys():
                    if self.config.io_overrides[model_name] != "Input":
                        current_input = self.interim_data[
                            self.config.io_overrides[model_name]
                        ]
            elif count > 0:
                current_input = out

            if model_name in self.models_nn.keys():
                cur_model = self.models_nn[model_name]
                out = cur_model(current_input)
            elif model_name in self.models_physics.keys():
                cur_model = self.models_physics[model_name]
                out = cur_model(
                    current_input,
                    self.config.models[model_name].forward_func,
                    self.config.models[model_name].jacobian_func,
                )
            if self.config.io_overrides:
                if model_name in self.config.io_overrides.keys():
                    self.interim_data[model_name] = out
        if self.config.io_overrides:
            output = 0
            for i in self.outputs:
                output += self.interim_data[i]
        else:
            output = out

        return output
