"""Ensemble module: HybridModel combining neural networks and physics solvers."""

import torch
import torch.nn
from adeptml import configs, models


class HybridModel(torch.nn.Module):
    """Torch module for serial hybrid physics-informed models.

    Combines neural network modules (MLPs, custom ``torch.nn.Module`` subclasses)
    with non-differentiable physics models in a single differentiable computation
    graph.  Models are executed in the insertion order of ``config.models``.

    Parameters
    ----------
    config : HybridConfig
        Configuration object specifying all sub-models and optional input
        routing.  See :class:`~adeptml.configs.HybridConfig` for details.

    Examples
    --------
    Serial NN → physics pipeline::

        cfg = HybridConfig(models={"nn": mlp_cfg, "physics": phy_cfg})
        model = HybridModel(cfg)
        out = model(x, phy_args=[bc, ic, sim_time])

    Custom input routing — pass original input to the physics model regardless
    of what the NN outputs::

        cfg = HybridConfig(
            models={"nn": mlp_cfg, "physics": phy_cfg},
            model_inputs={"physics": {"Input": None, "nn": None}},
        )
    """

    def __init__(self, config: configs.HybridConfig):
        super().__init__()
        self.models_nn = torch.nn.ModuleDict()
        self.models_physics = {}
        self.config = config

        # Resolve the apply function for each physics model once at init time.
        for model_name, model_cfg in config.models.items():
            if isinstance(model_cfg, torch.nn.Module):
                self.models_nn[model_name] = model_cfg.to(configs.DEVICE)
            elif isinstance(model_cfg, configs.PhysicsConfig):
                if model_cfg.use_split_vjp:
                    self.models_physics[model_name] = models.Physics_SplitVJP.apply
                elif model_cfg.use_vjp:
                    self.models_physics[model_name] = models.Physics_VJP.apply
                else:
                    self.models_physics[model_name] = models.Physics.apply
            elif isinstance(model_cfg, configs.MLPConfig):
                self.models_nn[model_name] = models.MLP(model_cfg).to(configs.DEVICE)

        # Compute the set of intermediate outputs that need to be cached for
        # custom input routing.  Done once here — not inside forward().
        self.to_save = []
        if config.model_inputs:
            to_save = []
            for _, vals in config.model_inputs.items():
                to_save += list(vals.keys())
            self.to_save = list(set(to_save))

    def forward(self, x, phy_args=None):
        """Run inference on the hybrid model.

        :param torch.Tensor x: Input tensor ``(batch, in_dim)``.
        :param phy_args: Extra positional arguments forwarded to physics
            sub-models (e.g. ``[bc, ic, sim_time]``).  May be a list of
            tensors or ``None``.
        :return: Output of the final model in the pipeline.
        :rtype: torch.Tensor
        """
        interim_data = {"Input": x}
        current_input = x
        out = x  # default if config.models is empty

        for model_name, model_cfg in self.config.models.items():
            # --- resolve input for this model ---
            if self.config.model_inputs and model_name in self.config.model_inputs:
                input_tensors = []
                for src_name, dims in self.config.model_inputs[model_name].items():
                    src = interim_data[src_name]
                    input_tensors.append(src[:, dims] if dims else src)
                current_input = torch.hstack(input_tensors)

            # --- run the model ---
            if model_name in self.models_nn:
                out = self.models_nn[model_name](current_input)
            elif model_name in self.models_physics:
                apply_fn = self.models_physics[model_name]
                if isinstance(model_cfg, configs.PhysicsConfig) and model_cfg.use_split_vjp:
                    # Split-VJP: forward_func already bakes in the pullback; no jacobian_func.
                    out = apply_fn(current_input, model_cfg.forward_func, phy_args)
                else:
                    out = apply_fn(
                        current_input,
                        model_cfg.forward_func,
                        model_cfg.jacobian_func,
                        phy_args,
                    )
            else:
                out = current_input

            # --- cache output for downstream routing ---
            if model_name in self.to_save:
                interim_data[model_name] = out

            current_input = out

        return out
