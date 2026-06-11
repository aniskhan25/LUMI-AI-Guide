import torch

from torch.profiler import profile, ProfilerActivity
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FakeData
from torchvision.models import vit_b_16

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


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=3):
    # note that "cuda" is used as a general reference to GPUs,
    # even when running on AMD GPUs that use ROCm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):

        prof = None
        if epoch == 1:  # profile the second epoch only
            print("Starting profile...")
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.start()

        model.train()
        running_loss = 0.0
        total_iterations = len(train_loader)
        max_iterations = int(total_iterations * 0.10)  # profile 10% of batches
        for i, (images, labels) in enumerate(train_loader):
            if i >= max_iterations:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if prof:
            prof.stop()
            prof.export_chrome_trace("trace.json")
            print("Trace saved to trace.json")

        print(f"Epoch {epoch+1}, Loss: {running_loss/max(len(train_loader), 1)}")

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


full_dataset = FakeData(
    size=2048,
    image_size=(3, 224, 224),
    num_classes=200,
    transform=transform,
)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=7)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=7)

train_model(model, criterion, optimizer, train_loader, val_loader)

torch.save(model.state_dict(), "vit_b_16_imagenet.pth")
