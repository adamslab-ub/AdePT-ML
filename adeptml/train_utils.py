import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from adeptml import configs
from adeptml import HybridModel


def train_step(
    model: HybridModel, optimizer: torch.optim.Optimizer, loss_fn, scheduler=None
):
    # Builds function that performs a step in the train loop
    def train_step(x, y, args, test=False):
        if not test:
            yhat = model.forward(x, args)
            loss = loss_fn(yhat, y)  # torch.mean(torch.abs(yhat-y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
        else:
            a = model.eval()
            with torch.no_grad():
                yhat = a(x, args)
                loss = loss_fn(yhat, y)
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_fn,
    scheduler,
    filename,
    epochs,
    print_training_loss=True,
    save_frequency=50,
):
    """Training Function.

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

    print_training_loss: bool
        Option to toggle printing epoch loss.

    save_frequency: int
        Number of epochs per which to save the model parameters to disk.


    Returns
    -------
        Trained Hybrid Model.
    """

    data_dir = os.path.join(os.getcwd(), f"Training_History_{filename}")
    if not os.path.exists(data_dir):
        os.system("mkdir %s" % data_dir)
    try:
        runs = max(
            [int(f.name.split("_")[-1]) for f in os.scandir(data_dir) if f.is_dir()]
        )
    except:
        runs = 0
    current_data_dir = f"{data_dir}/run_{runs+1}"
    cur_settings = ""
    # for i in asdict(model.config) :
    with SummaryWriter(log_dir=current_data_dir) as writer:
        train_step_obj = train_step(model, optimizer, loss_fn, scheduler)
        for epoch in range(epochs):
            train_batch_losses = []
            for data in train_loader:
                x_batch = data[0]
                y_batch = data[1]
                if len(data) > 2:
                    args = data[2:]
                else:
                    args = None
                loss = train_step_obj(x_batch, y_batch, args)
                train_batch_losses.append(loss)
            writer.add_scalar("Loss/train", np.mean(train_batch_losses), epoch)
            test_batch_losses = []
            for data in test_loader:
                x_batch = data[0]
                y_batch = data[1]
                if len(data) > 2:
                    args = data[2:]
                else:
                    args = None
                loss = train_step_obj(x_batch, y_batch, args, test=True)
                test_batch_losses.append(loss)
            writer.add_scalar("Loss/test", np.mean(test_batch_losses), epoch)
            if print_training_loss:
                print(
                    f"Train Loss {np.mean(train_batch_losses)} Test Loss {np.mean(test_batch_losses)}"
                )
            if epoch % save_frequency == 0:
                torch.save(
                    model.state_dict(), "%s/Model_%d.pt" % (current_data_dir, epoch)
                )
        torch.save(model.state_dict(), current_data_dir + "/Model_final.pt")

    return model
