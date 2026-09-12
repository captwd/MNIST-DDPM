import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST_ROOT = REPO_ROOT

TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
])


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_mnist_loaders(batch_size=64, num_workers=0, limit_train=None, limit_test=None):
    train_ds = torchvision.datasets.MNIST(
        root=MNIST_ROOT, train=True, download=True, transform=TRANSFORM)
    test_ds = torchvision.datasets.MNIST(
        root=MNIST_ROOT, train=False, download=True, transform=TRANSFORM)
    if limit_train is not None:
        train_ds = torch.utils.data.Subset(train_ds, range(min(limit_train, len(train_ds))))
    if limit_test is not None:
        test_ds = torch.utils.data.Subset(test_ds, range(min(limit_test, len(test_ds))))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, train_ds, test_ds


def load_mnist_arrays():
    train_ds = torchvision.datasets.MNIST(root=MNIST_ROOT, train=True, download=True)
    test_ds = torchvision.datasets.MNIST(root=MNIST_ROOT, train=False, download=True)

    def to_xy(ds):
        x = ds.data.numpy().astype(np.float32).reshape(len(ds), -1) / 255.0
        x = (x - 0.5) / 0.5
        y = ds.targets.numpy().astype(np.int64)
        return x, y

    x_train, y_train = to_xy(train_ds)
    x_test, y_test = to_xy(test_ds)
    return x_train, y_train, x_test, y_test


def torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_synthetic_dataset(path):
    blob = torch_load(path)
    return TensorDataset(blob['images'].float(), blob['labels'].long()), blob.get('meta', {})
