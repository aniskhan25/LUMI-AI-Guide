#!/bin/bash
#SBATCH --job-name=ramfs-demo
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh

: "${CONTAINER:?Set CONTAINER in env.sh}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT in env.sh}"

OUT_DIR="$SCRATCH_ROOT/ramfs-demo"
mkdir -p "$OUT_DIR"

# Pattern: run inside /tmp (RAMfs), copy outputs out before job ends
srun singularity exec "$CONTAINER" bash -c "
  set -euo pipefail
  python /dev/stdin <<'PY'
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FakeData
from torchvision.models import vit_b_16

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = FakeData(size=512, image_size=(3, 224, 224), num_classes=10, transform=transform)
train_ds, val_ds = random_split(dataset, [410, 102])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=7)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=7)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vit_b_16(weights=None).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} done')

# Save to /tmp (RAMfs) first
torch.save(model.state_dict(), '/tmp/vit_ramfs_demo.pth')
print('Saved to /tmp')
PY
  # Copy output from RAMfs to persistent storage before job ends
  cp /tmp/vit_ramfs_demo.pth '$OUT_DIR/vit_ramfs_demo.${SLURM_JOB_ID}.pth'
  echo 'Checkpoint copied to $OUT_DIR'
"
