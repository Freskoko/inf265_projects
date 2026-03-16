import torch
import numpy as np
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
from itertools import product
import math
import hashlib
import io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import patches
from matplotlib.patches import Patch
from torchvision import transforms
import json
from torch.utils.data import TensorDataset
import matplotlib.gridspec as gridspec

def plot_class_counter(all_targets, save_dir):

    max_target = max(all_targets)
    min_target = min(all_targets)

    counts = Counter(all_targets)
    plt.bar(
        list(counts.keys()), list(counts.values()), label=counts.keys()
    )
    plt.xlabel("Class")
    plt.ylabel("Count per class")
    plt.title("Class distribution in train set")
    for x, y in counts.items():
        plt.text(x, y, f"{y}", ha="center")

    plt.xticks([i for i in range(min_target,max_target+1)])
    plt.savefig(save_dir / "class_dist.png")
    plt.show()


def average_pixel_value(all_pixels, save_dir):
    plt.figure(figsize=(8,5))
    plt.hist(all_pixels.numpy(), bins=50, color='blue', edgecolor='black', density=True)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.title("Pixel value distribution in train set")
    plt.tight_layout()
    plt.savefig(save_dir / "pixel_dist.png")
    plt.show()


def pixels_outside_inside_hist(inside, outside, save_dir, label=""):
    plt.figure(figsize=(8,5))
    plt.hist(outside.numpy(), bins=50, alpha=0.6, color='blue', label='outside bbox', density=True)
    plt.hist(inside.numpy(), bins=50, alpha=0.6, color='green', label='inside bbox', density=True)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.title(f"Pixel value distribution for pixels inside/outside of bb (train set) {label}")
    plt.legend(handles=[
        Patch(color='blue', alpha=0.6, label='outside bbox'),
        Patch(color='green', alpha=0.6, label='inside bbox'),
        Patch(color='teal', alpha=0.6, label='overlap'),  # blue+green overlap = teal
    ])
    plt.tight_layout()
    plt.savefig(save_dir / f"inside_outside_pixel_dist_{label}.png")
    plt.show()