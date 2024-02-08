import os
import torch
from dataclasses import asdict
from joblib import load
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from hypyml import configs
from hypyml import HybridModel


def train_step(
    model: HybridModel, optimizer: torch.optim.Optimizer, loss_fn, scheduler=None
):
    # Builds function that performs a step in the train loop
    def train_step(x, y, test=False):
        if not test:
            yhat = model.forward(x)
            loss = loss_fn(yhat, y)  # torch.mean(torch.abs(yhat-y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
        else:
            a = model.eval()
            with torch.no_grad():
                yhat = a(x)
                loss = loss_fn(yhat, y)
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def train(
    model, train_loader, test_loader, optimizer, loss_fn, scheduler, filename, epochs
):
    """
    Training Function.

    Parameters
    ----------
    train_loader : torch.Torch_Dataloader
        Torch Dataloader with training samples.

    test_loader : torch.Torch_Dataloader
        Torch Dataloader with validation samples.

    optimizer : torch.optim.Optimizer
        Initialized Torch Optimizer.

    loss_fn : callable
        Loss function for training.

    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.

    filename : str
        File name for saving the trained model.

    epochs : int
        Number of training epochs.

    Returns
    -------
        Trained Hybrid Model.
    """

    data_dir = os.path.join(os.getcwd(), f"Training_History_{filename}")
    try:
        runs = max(
            [int(f.name.split("_")[-1]) for f in os.scandir(data_dir) if f.is_dir()]
        )
    except:
        runs = 0
    if not os.path.exists(data_dir):
        os.system("mkdir %s" % data_dir)
    with SummaryWriter(log_dir=f"{data_dir}/run_{runs+1}") as writer:
        train_step_obj = train_step(model, optimizer, loss_fn, scheduler)
        for epoch in range(epochs):
            train_batch_losses = []
            for x_batch, y_batch in train_loader:
                loss = train_step_obj(x_batch, y_batch)
                train_batch_losses.append(loss)
            writer.add_scalar("Loss/train", np.mean(train_batch_losses), epoch)
            test_batch_losses = []
            for x_batch, y_batch in test_loader:
                loss = train_step_obj(x_batch, y_batch, test=True)
                test_batch_losses.append(loss)
            writer.add_scalar("Loss/test", np.mean(test_batch_losses), epoch)
            print(
                f"Train Loss {np.mean(train_batch_losses)} Test Loss {np.mean(test_batch_losses)}"
            )
            if epoch % 50 == 0:
                torch.save(model.state_dict(), "%s/Model_%d.pt" % (data_dir, epoch))
        torch.save(model.state_dict(), data_dir + "/Model_final.pt")

    return model
