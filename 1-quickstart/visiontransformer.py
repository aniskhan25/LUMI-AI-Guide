import os
import sys
import torch

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FakeData
from torchvision.models import vit_b_16

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from resources.hdf5_dataset import HDF5Dataset

HDF5_PATH = os.environ.get("TINY_HDF5_PATH", "../resources/train_images.hdf5")
HDF5_FALLBACK = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "resources", "train_images.hdf5"
    )
)

# Define transformations for dataset
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = vit_b_16(weights="DEFAULT")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    # note that "cuda" is used as a general reference to GPUs,
    # even when running on AMD GPUs that use ROCm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}.")
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}%")


def run_training(full_train_dataset):
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=7)
    train_model(model, criterion, optimizer, train_loader, val_loader)


def get_hdf5_path():
    for path in (HDF5_PATH, HDF5_FALLBACK):
        if os.path.isfile(path):
            return path
    return None


hdf5_path = get_hdf5_path()
if hdf5_path is None:
    print("HDF5 not found; using FakeData.")
    fake_dataset = FakeData(
        size=2048,
        image_size=(3, 224, 224),
        num_classes=200,
        transform=transform,
    )
    run_training(fake_dataset)
else:
    print(f"Using HDF5 dataset: {hdf5_path}")
    with HDF5Dataset(hdf5_path, transform=transform) as full_train_dataset:
        run_training(full_train_dataset)

torch.save(model.state_dict(), "vit_b_16_imagenet.pth")
