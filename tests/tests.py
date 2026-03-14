import torch
import unittest
import numpy as np
from adeptml import models, configs, HybridModel, train


# ---------------------------------------------------------------------------
# Shared physics helpers
# ---------------------------------------------------------------------------

def jac_fun(x):
    """Full Jacobian of f(x) = 2x: shape (batch, out, in) = (batch, n, n)."""
    return 2 * np.array([np.diag(x[i, :].ravel()) for i in range(x.shape[0])])


def vjp_fun(x, g):
    """VJP of f(x) = x**2: df/dx * g = 2x * g, shape (batch, n)."""
    return 2 * x * g


def split_vjp_func(x):
    """Split-VJP forward for f(x) = x**2.

    Returns (output, pullback_fn) mimicking the jax.vjp pattern.
    All arrays are numpy — conversion is handled by Physics_SplitVJP.
    """
    out = x ** 2

    def pullback(g):
        # Returns a tuple of cotangents, one per argument of split_vjp_func.
        # Index 0 corresponds to x.
        return (2 * x * g,)

    return out, pullback


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestADEPTML(unittest.TestCase):

    # --- MLP -----------------------------------------------------------------

    def test_MLP(self):
        mlp_config = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=5,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        model = models.MLP(mlp_config).to(configs.DEVICE)
        x = torch.rand((4, 2)).to(configs.DEVICE)
        y = model(x)
        self.assertEqual(y.shape[-1], 5)

    # --- Physics (Jacobian mode) ---------------------------------------------

    def test_Physics_forward(self):
        """Physics.apply produces the correct output shape."""
        phy_config = configs.PhysicsConfig(
            forward_func=lambda x: x ** 2,
            jacobian_func=jac_fun,
        )
        x = torch.rand((4, 2)).to(configs.DEVICE)
        y = models.Physics.apply(x, phy_config.forward_func, phy_config.jacobian_func)
        self.assertEqual(y.shape, x.shape)

    def test_Physics_gradient_flow(self):
        """Gradient flows back through Physics (Jacobian mode)."""
        phy_config = configs.PhysicsConfig(
            forward_func=lambda x: x ** 2,
            jacobian_func=jac_fun,
        )
        x = torch.rand((4, 2), requires_grad=True).to(configs.DEVICE)
        y = models.Physics.apply(x, phy_config.forward_func, phy_config.jacobian_func)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        # d/dx sum(x**2) = 2x
        self.assertTrue(torch.allclose(x.grad, 2 * x.detach(), atol=1e-5))

    # --- Physics_VJP ---------------------------------------------------------

    def test_Physics_VJP_forward(self):
        """Physics_VJP.apply produces the correct output shape."""
        phy_config = configs.PhysicsConfig(
            forward_func=lambda x: x ** 2,
            jacobian_func=vjp_fun,
            use_vjp=True,
        )
        x = torch.rand((4, 2)).to(configs.DEVICE)
        y = models.Physics_VJP.apply(
            x, phy_config.forward_func, phy_config.jacobian_func
        )
        self.assertEqual(y.shape, x.shape)

    def test_Physics_VJP_gradient_flow(self):
        """Gradient flows back through Physics_VJP and is numerically correct."""
        phy_config = configs.PhysicsConfig(
            forward_func=lambda x: x ** 2,
            jacobian_func=vjp_fun,
            use_vjp=True,
        )
        x = torch.rand((4, 2), requires_grad=True).to(configs.DEVICE)
        y = models.Physics_VJP.apply(
            x, phy_config.forward_func, phy_config.jacobian_func
        )
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.allclose(x.grad, 2 * x.detach(), atol=1e-5))

    # --- Physics_SplitVJP ----------------------------------------------------

    def test_Physics_SplitVJP_forward(self):
        """Physics_SplitVJP.apply produces the correct output shape."""
        phy_config = configs.PhysicsConfig(
            forward_func=split_vjp_func,
            use_split_vjp=True,
        )
        x = torch.rand((4, 2)).to(configs.DEVICE)
        y = models.Physics_SplitVJP.apply(x, phy_config.forward_func)
        self.assertEqual(y.shape, x.shape)

    def test_Physics_SplitVJP_gradient_flow(self):
        """Gradient flows through Physics_SplitVJP via the stored pullback closure."""
        phy_config = configs.PhysicsConfig(
            forward_func=split_vjp_func,
            use_split_vjp=True,
        )
        x = torch.rand((4, 2), requires_grad=True).to(configs.DEVICE)
        y = models.Physics_SplitVJP.apply(x, phy_config.forward_func)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        # d/dx sum(x**2) = 2x
        self.assertTrue(torch.allclose(x.grad, 2 * x.detach(), atol=1e-5))

    def test_Physics_SplitVJP_no_jacobian_func_required(self):
        """PhysicsConfig with use_split_vjp=True does not require jacobian_func."""
        # Should construct without error
        phy_config = configs.PhysicsConfig(
            forward_func=split_vjp_func,
            use_split_vjp=True,
        )
        self.assertIsNone(phy_config.jacobian_func)

    # --- HybridModel wiring --------------------------------------------------

    def test_use_vjp_wiring(self):
        """use_vjp=True in PhysicsConfig routes to Physics_VJP.apply in HybridModel."""
        phy_config = configs.PhysicsConfig(
            forward_func=lambda x: x ** 2,
            jacobian_func=vjp_fun,
            use_vjp=True,
        )
        config = configs.HybridConfig(models={"Physics": phy_config})
        model = HybridModel(config)
        self.assertIs(model.models_physics["Physics"].__self__, models.Physics_VJP)

    def test_use_split_vjp_wiring(self):
        """use_split_vjp=True routes to Physics_SplitVJP.apply in HybridModel."""
        phy_config = configs.PhysicsConfig(
            forward_func=split_vjp_func,
            use_split_vjp=True,
        )
        config = configs.HybridConfig(models={"Physics": phy_config})
        model = HybridModel(config)
        self.assertIs(model.models_physics["Physics"].__self__, models.Physics_SplitVJP)

    def test_use_split_vjp_takes_precedence_over_use_vjp(self):
        """use_split_vjp=True takes precedence over use_vjp=True."""
        phy_config = configs.PhysicsConfig(
            forward_func=split_vjp_func,
            use_vjp=True,
            use_split_vjp=True,
        )
        config = configs.HybridConfig(models={"Physics": phy_config})
        model = HybridModel(config)
        self.assertIs(model.models_physics["Physics"].__self__, models.Physics_SplitVJP)

    # --- Ensemble: serial and hybrid -----------------------------------------

    def test_ensemble_serial(self):
        phy_1 = configs.PhysicsConfig(
            forward_func=lambda x: x ** 2, jacobian_func=jac_fun
        )
        mlp_1 = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=5,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        config = configs.HybridConfig(models={"Physics_1": phy_1, "MLP_1": mlp_1})
        model = HybridModel(config)
        x = torch.rand((4, 2)).to(configs.DEVICE)
        y = model(x).detach().cpu().numpy()
        self.assertEqual(y.shape[1], 5)

    def test_ensemble_hybrid(self):
        phy_1 = configs.PhysicsConfig(
            forward_func=lambda x: x ** 2, jacobian_func=jac_fun
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
        self.assertEqual(y.shape[1], 2 * x.shape[1])

    def test_ensemble_split_vjp_gradient_flows_to_nn(self):
        """Gradient flows from Physics_SplitVJP back through NN parameters."""
        mlp_config = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=4,
            num_hidden_layers=1,
            num_output_dim=2,
            hidden_activation="leakyrelu",
            output_activation="tanh",
        )
        phy_config = configs.PhysicsConfig(
            forward_func=split_vjp_func,
            use_split_vjp=True,
        )
        config = configs.HybridConfig(
            models={"MLP": mlp_config, "Physics": phy_config}
        )
        model = HybridModel(config)
        x = torch.rand((4, 2))
        out = model(x)
        out.sum().backward()
        # All NN parameters should have gradients
        for name, param in model.models_nn.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"No grad for {name}")

    def test_custom_module(self):
        """Custom torch.nn.Module instances are accepted in HybridConfig."""
        class CustModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x ** 2 + 3 * x

        mlp_1 = configs.MLPConfig(
            num_input_dim=10,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=5,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        config = configs.HybridConfig(
            models={"MLP_1": mlp_1, "Custom_Physics": CustModule()},
        )
        model = HybridModel(config).to(configs.DEVICE)
        x = torch.rand((4, 10)).to(configs.DEVICE)
        output = model(x)
        self.assertEqual(output.shape[-1], 5)

    # --- Training loop -------------------------------------------------------

    def test_train_serial(self):
        mlp_config = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=2,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        phy_config = configs.PhysicsConfig(
            forward_func=lambda x, z: 4 * x,
            jacobian_func=lambda x, z: 4 * np.ones((x.shape[0], 2, x.shape[1])),
        )
        config = configs.HybridConfig(models={"MLP": mlp_config, "Physics": phy_config})
        model = HybridModel(config).to(configs.DEVICE)

        x_train = torch.rand(4, 2).to(configs.DEVICE)
        y_train = torch.rand(4, 2).to(configs.DEVICE)
        z_train = torch.rand(4, 2).to(configs.DEVICE)
        x_test = torch.rand(4, 2).to(configs.DEVICE)
        y_test = torch.rand(4, 2).to(configs.DEVICE)
        z_test = torch.rand(4, 2).to(configs.DEVICE)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train, z_train),
            batch_size=2, shuffle=True, drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test, z_test),
            batch_size=2, shuffle=True, drop_last=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95)
        model = train(
            model, train_loader, test_loader,
            optimizer, torch.nn.MSELoss(), scheduler,
            "Test/Test", epochs=2, print_training_loss=False,
        )
        self.assertIsInstance(model, HybridModel)

    def test_train_with_grad_clip(self):
        """Training completes without error when grad_clip is set."""
        mlp_config = configs.MLPConfig(
            num_input_dim=2,
            num_hidden_dim=2,
            num_hidden_layers=2,
            num_output_dim=2,
            hidden_activation="leakyrelu",
            output_activation="leakyrelu",
        )
        config = configs.HybridConfig(models={"MLP": mlp_config})
        model = HybridModel(config).to(configs.DEVICE)

        x = torch.rand(4, 2).to(configs.DEVICE)
        y = torch.rand(4, 2).to(configs.DEVICE)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=2,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model = train(
            model, loader, loader,
            optimizer, torch.nn.MSELoss(), None,
            "Test/GradClip", epochs=2,
            print_training_loss=False, grad_clip=1.0,
        )
        self.assertIsInstance(model, HybridModel)


if __name__ == "__main__":
    unittest.main()
