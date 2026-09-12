import glob
import os

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset

    Supports two data formats:
      1. PNG images organized as ``<im_path>/<label>/*.png`` (original behavior).
      2. Raw MNIST IDX binary files (``*-ubyte.gz``), loaded via
         ``torchvision.datasets.MNIST``.  When ``im_path`` does not exist the
         dataset automatically falls back to this mode; ``<root>`` for
         torchvision is inferred as the directory two levels above ``im_path``
         (e.g. ``data/train/images`` -> ``data``), which is where the
         ``MNIST/raw/*.gz`` files are expected / auto-downloaded to.
    """
    def __init__(self, split, im_path, im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images, or fallback root hint
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_ext = im_ext
        self._tv_dataset = None  # torchvision MNIST dataset (fallback mode)
        self.images = None
        self.labels = None

        if os.path.isdir(im_path):
            self.images, self.labels = self.load_images(im_path)
        else:
            print('[MnistDataset] PNG directory "{}" not found; falling back '
                  'to torchvision.datasets.MNIST (IDX binary format).'.format(im_path))
            self._tv_dataset = self._load_torchvision_mnist(im_path, split)

    @staticmethod
    def _load_torchvision_mnist(im_path, split):
        """
        Load MNIST from the canonical IDX binary files, using the parent
        directory of ``im_path`` as the ``root`` argument for
        ``torchvision.datasets.MNIST``.  Binary files are automatically
        downloaded when missing.
        """
        # config/default.yaml uses ``data/train/images`` -> root = ``data``
        candidate = os.path.normpath(im_path)
        for _ in range(3):
            if os.path.basename(candidate) == '':
                candidate = os.path.dirname(candidate)
            base = os.path.basename(candidate).lower()
            if base in ('train', 'test'):
                candidate = os.path.dirname(candidate)
                break
            if base == 'images':
                candidate = os.path.dirname(candidate)
                break
            break
        root = candidate if candidate else 'data'
        # Prefer the MNIST/raw files already shipped in the repo root
        # (torchvision expects <root>/MNIST/raw), so no duplicate download.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isdir(os.path.join(repo_root, 'MNIST', 'raw')):
            root = repo_root
        else:
            root = os.path.abspath(root)
        is_train = split.lower() == 'train'
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # match the original dataset's [-1, 1] normalization exactly
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])
        ds = torchvision.datasets.MNIST(
            root=root, train=is_train, download=True, transform=transform,
        )
        print('[MnistDataset] Loaded {} {} samples from torchvision MNIST '
              '(root={}).'.format(len(ds), split, root))
        return ds

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels

    def __len__(self):
        if self._tv_dataset is not None:
            return len(self._tv_dataset)
        return len(self.images)

    def __getitem__(self, index):
        if self._tv_dataset is not None:
            im_tensor, _label = self._tv_dataset[index]
            return im_tensor
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)

        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
