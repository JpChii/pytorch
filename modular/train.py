"""
Trains a PyTorch image classification model using device-agnostic code
"""

import os
import torch
from torch import nn
from src.data.data_setup import create_dataloders
from src.data.get_data import get_data
from src.model.model_builder import TinyVGG
from src.model.engine import train_model
from src.helpers.utils import save_model

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Download data
images_dir = get_data(
    request_url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    data_path="data",
)

# Setup directories
train_dir = f"{images_dir}/train"
test_dir = f"{images_dir}/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = create_dataloders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE,    
)

# Create model with help from model_builder.py
model = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(class_names),
).to(device=device)

# Setup loss and optimzier
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=LEARNING_RATE,
)

# Start training with engine
train_model(
    model=model,
    train_dataloder=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

# Save model
save_model(
    model=model,
    target_dir="models/",
    model_name="tiny_vgg_food_classifier.pth"
)