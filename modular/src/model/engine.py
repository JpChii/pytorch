"""
Contains functions for training a model
"""

import torch
from torch import nn

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """_summary_
    Trains a PyTorch model for a single epoch

    Turna a PyTorch model into a training mode and then runs through
    forward pass, loss calculation, backward pass, optimizer step and tracks stats

    Args:
        model (nn.Module): A PyTorch model to be trained
        dataloader (torch.utils.data.DataLoader): A PyTorch dataloader to be trained on
        loss_fn (nn.Module): A PyTorch loss function to minimize
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
        Tuple[float, float]: A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy).

        For example:
        (0.112,0.8734)
    """

    # Put model in train mode
    model.train()

    # Loss, accuracy
    train_loss, train_accuracy = 0, 0

    # Loop through data batches
    for x, y in dataloader:
        # Send data to target device
        x, y = x.to(device), y.to(device)

        # Forward pass
        y_pred = model(x)

        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimize step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_accuracy += (y_pred_class == y).sum().item() / len(y_pred)

    # Average of loss and accuracy with batches
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """_summary_

    Args:
        model (nn.Module): A Pytorch model to be tested
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on
        loss_fn (nn.Module): A PyTorch loss function to calculate loss on test data
        device (torch.device): A target device to compute on (e.g "cuda" or "cpu")

    Returns:
        A tuple of test loss and test accuracy metrics.
        In the form (test_loss, test_accuracy). For example

        (0.0223, 0.8985)
    """

    # Put model in eval mode
    model.eval()

    # Loss, accuracy
    test_loss, test_accuracy = 0, 0

    # Turn on intefernce context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for x, y in dataloader:
            # Send data to targetd evice
            x, y = x.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(x)

            # Calculate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy metric across all batches
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()) / len(test_pred_labels)

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_accuracy = test_accuracy / len(dataloader)
        return test_loss, test_accuracy


def train_model(
    model: nn.Module,
    train_dataloder: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List]:
    """_summary_
    Trains and tests a PyTorch model

    Passes a target PyTorch models througb train_step() and test_step()
    functions for a number of epochs, training and testing the model in the same epoch loop.

    Calculates, prints and tores evaluation metrics throughout.

    Args:
        model (nn.Module): A PyTorch model to be trained and tested
        train_dataloder (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        loss_fn (nn.Module): A PyTorch loss function to calculate loss on test datasets
        epochs (int): An integer number of epochs to train the model
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
        A dictionary of training and testeing loss as wellas training and
        testing accuracy metrics. Each metric has a value in a list for each epoch.
        In the form: {
        train_loss: [...],
        train_acc: [...],
        test_loss: [...],
        test_loss: [...]
        }

        For example for epochs=2:{
        train_loss: [1, 2],
        train_acc: [1, 2],
        test_loss: [1, 2],
        test_acc: [1, 2]
        }
    """

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    # Loop through training and testing steps for a number of epoochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloder,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of epochs
    return results
