import os
import torch
import psutil
import mlflow

import torch.distributed as dist
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import FakeData
from torchvision.models import vit_b_16

dist.init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
rank = int(os.environ["RANK"])


def set_cpu_affinity(local_rank):
    # Mapping from GCD to closest CPU cores on a LUMI-G node.
    # See https://docs.lumi-supercomputer.eu/hardware/lumig/
    LUMI_GPU_CPU_map = {
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    psutil.Process().cpu_affinity(LUMI_GPU_CPU_map[local_rank])


set_cpu_affinity(local_rank)

if rank == 0:
    mlflow.set_tracking_uri(os.environ["PWD"] + "/mlruns")
    mlflow.start_run(run_name=os.getenv("SLURM_JOB_ID", "local"))

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = vit_b_16(weights="DEFAULT").to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
            mlflow.log_metric("loss", running_loss / len(train_loader), step=epoch)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if rank == 0:
            print(f"Accuracy: {100 * correct / total}%")
            mlflow.log_metric("accuracy", correct / total, step=epoch)


full_dataset = FakeData(size=2048, image_size=(3, 224, 224), num_classes=200, transform=transform)
train_size = int(0.8 * len(full_dataset))
train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset), batch_size=32, num_workers=7)
val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset), batch_size=32, num_workers=7)

train_model(model, criterion, optimizer, train_loader, val_loader)
dist.destroy_process_group()

if rank == 0:
    mlflow.end_run()
    torch.save(model.state_dict(), "vit_b_16_imagenet.pth")
