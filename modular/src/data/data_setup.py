"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Number of workers for dataloader
NUM_WORKERS = os.cpu_count()


def create_dataloders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """_summary_

    Takes in a training directory and testing directory paths and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir (str): Train data directory
        test_dir (str): Test data directory
        transform (transforms.Compose): Transform to use with dataset
        batch_size (int): Batch Size for dataloder
        num_workers (int, optional): Number of workers for dataloder. Defaults to NUM_WORKERS.

    Returns:
        A tuple of (train_dataloder, test_dataloader, class_names).
        where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloders(
            train_dir, test_dir, transform, batch_size, num_workers
        )
    """

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.class_names

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
