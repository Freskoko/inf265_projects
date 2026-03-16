import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from math import ceil, floor
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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
import json
from torch.utils.data import TensorDataset
from constants import IMG_HEIGHT, IMG_WIDTH

def turn_no_class_to_nans(sample):
    img, vec = sample
    if vec[0] == 0:
        vec[5] = -1 # no class
        vec[1:5] = np.nan # i just did this lets see if it works
    return img, vec


class Preprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, train_dataset):
        images, _ = train_dataset.tensors

        mean = images.mean(dim=(0, 2, 3))
        std = images.std(dim=(0, 2, 3))

        self.mean = mean
        self.std = std

    def process(self, dataset):
        images, labels = dataset.tensors
        images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return TensorDataset(images, labels)

    def unnormalize(self, img):
        img = img.clone()
        img = img * self.std[:, None, None] + self.mean[:, None, None]
        return torch.clamp(img, 0, 1)

def intersection_union(pred, true):
    # Unpack [z, x_center, y_center, width, height]
    # Convert to corners [x1, y1, x2, y2]
    def to_corners(box):
        xc, yc, w, h = box[1], box[2], box[3], box[4]
        return xc - w/2, yc - h/2, xc + w/2, yc + h/2

    p_x1, p_y1, p_x2, p_y2 = to_corners(pred)
    t_x1, t_y1, t_x2, t_y2 = to_corners(true)

    # Calculate intersection boundaries
    i_x1 = torch.max(p_x1, t_x1)
    i_y1 = torch.max(p_y1, t_y1)
    i_x2 = torch.min(p_x2, t_x2)
    i_y2 = torch.min(p_y2, t_y2)

    # Width and height of intersection (clamp to 0 if no overlap)
    inter_w = torch.clamp(i_x2 - i_x1, min=0)
    inter_h = torch.clamp(i_y2 - i_y1, min=0)
    intersection = inter_w * inter_h

    # Areas
    area_p = torch.clamp(p_x2 - p_x1, min=0) * torch.clamp(p_y2 - p_y1, min=0)
    area_t = torch.clamp(t_x2 - t_x1, min=0) * torch.clamp(t_y2 - t_y1, min=0)

    union = area_p + area_t - intersection

    return union.item(), intersection.item()


def scale_vars(x,y,w,h):
    """
    Scale vars to plot correctly
    """
    scaled_x = x*IMG_WIDTH
    scaled_w = w*IMG_WIDTH
    scaled_y = y*IMG_HEIGHT
    scaled_h = h*IMG_HEIGHT

    scaled_x = scaled_x - scaled_w / 2
    scaled_y = scaled_y - scaled_h / 2

    return scaled_x, scaled_y, scaled_w, scaled_h


def preprocess_out_noise(dataset, threshold=1.5):
    img = dataset[0].clone()
    img[img < threshold] = 0
    return (img, dataset[1])