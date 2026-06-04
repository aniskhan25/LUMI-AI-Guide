import os
import time
import torch
import psutil
import argparse
import deepspeed

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import FakeData
from torchvision.models import vit_b_16

parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

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
    cpu_list = LUMI_GPU_CPU_map[local_rank]
    print(f"Rank {rank} (local {local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)


set_cpu_affinity(local_rank)

deepspeed.init_distributed()

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


def train_model(args, model, criterion, optimizer, train_loader, val_loader, epochs=10):
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
    )

    if rank == 0:
        start = time.time()

    for epoch in range(epochs):
        model_engine.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
            optimizer.zero_grad()
            outputs = model_engine(images)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()
            running_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        model_engine.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
                outputs = model_engine(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if rank == 0:
            print(f"Accuracy: {100 * correct / total}%")

    if rank == 0:
        print(f"Time elapsed (s): {time.time()-start}")


full_dataset = FakeData(
    size=2048,
    image_size=(3, 224, 224),
    num_classes=200,
    transform=transform,
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32, num_workers=7)

val_sampler = DistributedSampler(val_dataset)
val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=32, num_workers=7)

train_model(args, model, criterion, optimizer, train_loader, val_loader)

if rank == 0:
    torch.save(model.state_dict(), "vit_b_16_imagenet.pth")
