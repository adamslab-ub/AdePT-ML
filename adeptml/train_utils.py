"""Training utilities: batch step helper and full training loop."""

import os
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from adeptml import HybridModel


def train_step(
    model: HybridModel,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    scheduler=None,
    grad_clip: Optional[float] = None,
):
    """Build and return a single-batch training step closure.

    Parameters
    ----------
    model : HybridModel
        The hybrid model to train.
    optimizer : torch.optim.Optimizer
        Initialized optimizer.
    loss_fn : callable
        Loss function ``(y_pred, y_true) -> scalar tensor``.
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler stepped after each batch.
    grad_clip : float, optional
        If set, gradients are clipped to this maximum L2 norm before the
        optimizer step.  Useful for physics-informed training where gradient
        spikes can occur.

    Returns
    -------
    callable
        ``_batch_step(x, y, args, test=False) -> float`` — runs one forward
        (and optionally backward) pass and returns the scalar loss.
    """

    def _batch_step(x, y, args, test=False):
        if not test:
            yhat = model.forward(x, args)
            loss = loss_fn(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
        else:
            with torch.no_grad():
                yhat = model.forward(x, args)
                loss = loss_fn(yhat, y)
        return loss.item()

    return _batch_step


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
    grad_clip: Optional[float] = None,
):
    """Full training loop with TensorBoard logging and checkpointing.

    Parameters
    ----------
    model : HybridModel
        The hybrid model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding ``(x, y)`` or ``(x, y, *args)`` batches.
    test_loader : torch.utils.data.DataLoader
        DataLoader yielding validation batches in the same format.
    optimizer : torch.optim.Optimizer
        Initialized optimizer.
    loss_fn : callable
        Loss function ``(y_pred, y_true) -> scalar tensor``.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.
    filename : str
        Base directory name for saving runs and TensorBoard logs.  A
        sub-directory ``run_N`` is created automatically (N auto-increments).
    epochs : int
        Number of training epochs.
    print_training_loss : bool
        Print epoch loss to stdout when ``True``.
    save_frequency : int
        Save a model checkpoint every this many epochs.
    grad_clip : float, optional
        Maximum L2 norm for gradient clipping.  ``None`` disables clipping.

    Returns
    -------
    HybridModel
        The trained model (same object, modified in place).
    """
    data_dir = os.path.join(os.getcwd(), filename)
    os.makedirs(data_dir, exist_ok=True)
    try:
        runs = max(
            int(f.name.split("_")[-1]) for f in os.scandir(data_dir) if f.is_dir()
        )
    except ValueError:
        runs = 0
    current_data_dir = f"{data_dir}/run_{runs + 1}"

    with SummaryWriter(log_dir=current_data_dir) as writer:
        step_fn = train_step(model, optimizer, loss_fn, scheduler, grad_clip)
        for epoch in range(epochs):
            t1 = time.time()

            train_batch_losses = []
            for data in train_loader:
                x_batch, y_batch = data[0], data[1]
                args = list(data[2:]) if len(data) > 2 else None
                train_batch_losses.append(step_fn(x_batch, y_batch, args))
            writer.add_scalar("Loss/train", np.mean(train_batch_losses), epoch)

            test_batch_losses = []
            for data in test_loader:
                x_batch, y_batch = data[0], data[1]
                args = list(data[2:]) if len(data) > 2 else None
                test_batch_losses.append(step_fn(x_batch, y_batch, args, test=True))
            writer.add_scalar("Loss/test", np.mean(test_batch_losses), epoch)

            t2 = time.time()
            if print_training_loss:
                print(
                    f"Epoch {epoch} | Time: {t2-t1:.2f}s | "
                    f"Train Loss: {np.mean(train_batch_losses):.6f} | "
                    f"Test Loss: {np.mean(test_batch_losses):.6f}"
                )

            if epoch % save_frequency == 0 and epoch != 0:
                torch.save(
                    model.state_dict(), f"{current_data_dir}/Model_{epoch}.pt"
                )

        torch.save(model.state_dict(), f"{current_data_dir}/model_final.pt")

    return model
