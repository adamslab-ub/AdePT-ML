import torch
import unittest
from hypyml import models
from hypyml import configs


class TestHyPyML(unittest.TestCase):
    def test_MLP(self):
        self.mlp_config = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=5,
            activation_functions="leakyrelu",
        )
        model = models.MLP(self.mlp_config)
        x = torch.rand((4, 2))
        y = model(x)
        assert y.shape[-1] == 5

    def test_Physics(self):
        self.phy_config = configs.PhysicsConfig(
            forward_func=lambda x: torch.sum(x**2, dim=1),
            jacobian_func=lambda x: torch.diag(2 * x),
        )
        model = models.Physics.apply
        x = torch.rand((4, 2))
        y = model(x, self.phy_config.forward_func, self.phy_config.jacobian_func)
        assert y.shape[0] == x.shape[0]
        pass

    def test_ensemble(self):
        config = configs.EnsembleConfig(
            models={
                "Transfer_Network": self.mlp_config,
                "Physics": self.phy_config,
                "Correction_Layers": self.mlp_config,
            },
            non_serial_inputs={"Correction_Layers": "Transfer_Network"},
        )

    def test_train(self):
        pass


if __name__ == "__main__":
    unittest.main()
