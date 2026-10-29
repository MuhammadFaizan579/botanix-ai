"""Re-evaluate the trained model on the held-out test split.

Usage:
    python src/evaluate.py

Downloads Oxford 102 Flowers (about 330 MB) on the first run, rebuilds the same
70/15/15 split used in training (seed 42) and reports accuracy and macro F1.
"""
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

from predict import TRANSFORM, load_model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Merge the three official splits (1020 + 1020 + 6149 = 8189 images)
full = ConcatDataset([
    datasets.Flowers102(root=str(DATA_DIR), split=s, transform=TRANSFORM, download=True)
    for s in ["train", "val", "test"]
])

# Same split as training: seed 42, 70% train / 15% val / 15% test
n = len(full)
g = torch.Generator().manual_seed(42)
idx = torch.randperm(n, generator=g).tolist()
n_train, n_val = int(0.70 * n), int(0.15 * n)
test_idx = idx[n_train + n_val:]

loader = DataLoader(Subset(full, test_idx), batch_size=32, shuffle=False, num_workers=2)

model = load_model().to(device)
preds, targets = [], []
with torch.no_grad():
    for x, y in loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        targets += y.tolist()

print(f"Test images:   {len(targets)}")
print(f"Test accuracy: {accuracy_score(targets, preds):.4f}")
print(f"Macro F1:      {f1_score(targets, preds, average='macro'):.4f}")
