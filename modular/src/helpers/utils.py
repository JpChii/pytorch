"""
Contains various utility functions for PyTorhc model trainin and saving
"""

import torch
from torch import nn
from pathlib import Path


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """_summary_
    Saves a PyTorch model in tarrget directory

    Args:
        model (torch.nn.Module): A targe PyTorch model to save
        target_dir (str): A directory to save the model to
        model_name (str): A filename for the saved model. Should include either ".pth" or ".pt" as the file extension

    Example usage:
        save_model(
            model=model_0,
            target_dir="models/",
            model_name="custom.pth"
        )
    """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create model save path
    assert model_name.endswith(".pth") or model.endswith(".pt")
    model_save_path = f"{target_dir_path}/{model_name}"

    # Save model state dict
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
