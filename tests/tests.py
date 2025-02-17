import torch
import unittest
import numpy as np
from adeptml import models, configs, HybridModel, train


def jac_fun(x):
    jacs = np.array([np.diag(x[i, :].ravel()) for i in range(x.shape[0])])
    print(jacs.shape)
    return 2 * jacs


class TestADEPTML(unittest.TestCase):
    def test_MLP(self):
        self.mlp_config = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=5,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        model = models.MLP(self.mlp_config).to(configs.DEVICE)
        x = torch.rand((4, 2)).to(configs.DEVICE)
        y = model(x)
        assert y.shape[-1] == 5

    def test_Physics(self):
        self.phy_config = configs.PhysicsConfig(
            forward_func=lambda x: np.sum(x**2, axis=1),
            jacobian_func=jac_fun,
        )
        model = models.Physics.apply
        x = torch.rand((4, 2)).to(configs.DEVICE)

        y = model(x, self.phy_config.forward_func, self.phy_config.jacobian_func)
        assert y.shape[0] == x.shape[0]
        pass

    def test_ensemble_serial(self):
        phy_1 = configs.PhysicsConfig(
            forward_func=lambda x: x**2, jacobian_func=jac_fun
        )
        mlp_1 = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=5,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        config = configs.HybridConfig(
            models={"Physics_1": phy_1, "MLP_1": mlp_1},
        )
        model = HybridModel(config)
        x = torch.rand((4, 2)).to(configs.DEVICE)

        y = model(x).detach().cpu().numpy()
        x = x.detach().cpu().numpy()

        assert y.shape[1] == 5

    def test_ensemble_hybrid(self):
        phy_1 = configs.PhysicsConfig(
            forward_func=lambda x: x**2,
            jacobian_func=jac_fun,
        )
        phy_2 = configs.PhysicsConfig(
            forward_func=lambda x: 10 * x, jacobian_func=jac_fun
        )
        config = configs.HybridConfig(
            models={"Physics_1": phy_1, "Physics_2": phy_2},
            model_inputs={"Physics_2": {"Input": None, "Physics_1": [0, 1]}},
        )
        model = HybridModel(config)
        x = torch.rand((4, 2))
        y = model(x).detach().cpu().numpy()
        assert y.shape[1] == 2 * x.shape[1]

    def test_custom_module(self):
        class cust_module(torch.nn.Module):
            def __init__(self):
                super(cust_module, self).__init__()

            def forward(self, x):
                return x**2 + 3 * x

        mlp_1 = configs.MLPConfig(
            num_input_dim=10,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=5,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        config = configs.HybridConfig(
            models={"MLP_1": mlp_1, "Custom_Physics": cust_module},
        )
        model = HybridModel(config).to(configs.DEVICE)
        input = torch.rand((4, 10)).to(configs.DEVICE)
        output = model(input)
        assert output.shape[-1] == 5

    def test_train_serial(self):
        mlp_config = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=2,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        phy_2 = configs.PhysicsConfig(
            forward_func=lambda x, y: 4 * x,
            jacobian_func=lambda x, y: 4 * np.ones((x.shape[0], 2, x.shape[1])),
        )
        config = configs.HybridConfig(
            models={"MLP": mlp_config, "Physics": phy_2},
        )
        model = HybridModel(config).to(configs.DEVICE)
        x_test = torch.rand(4, 2).to(configs.DEVICE)
        y_test = torch.rand(4, 2).to(configs.DEVICE)
        z_test = torch.rand(4, 2).to(configs.DEVICE)
        x_train = torch.rand(4, 2).to(configs.DEVICE)
        y_train = torch.rand(4, 2).to(configs.DEVICE)
        z_train = torch.rand(4, 2).to(configs.DEVICE)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test, z_test),
            batch_size=2,
            shuffle=True,
            drop_last=True,
        )
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train, z_train),
            batch_size=2,
            shuffle=True,
            drop_last=True,
        )
        lr = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=30, gamma=0.95
        )
        loss_fn = torch.nn.MSELoss()
        print("newtest")
        model = train(
            model,
            train_loader,
            test_loader,
            optimizer,
            loss_fn,
            scheduler,
            "Test/Test",
            2,
        )


if __name__ == "__main__":
    unittest.main()
