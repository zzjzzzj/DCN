# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from PIL import Image
import numpy as np
import importlib.util


class MyFashionMNIST(Dataset):
    """
    Minimal Fashion-MNIST loader that mirrors CIFAR wrapper behavior.
    Returns (img, target, not_aug_img) when `return_not_aug=True` (train),
    else returns (img, target).
    """
    def __init__(self, root: str, train: bool = True, transform=None,
                 target_transform=None, return_not_aug: bool = True) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.return_not_aug = return_not_aug

        # ToTensor for the not-augmented copy; resize to 32 and channel replication handled in transforms
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])

        # Load mnist_reader dynamically from the provided fashion_mnist_master
        fm_root = os.path.join(os.path.dirname(__file__), 'fashion_mnist_master')
        mnist_path = os.path.join(fm_root, 'utils', 'mnist_reader.py')
        if not os.path.exists(mnist_path):
            raise FileNotFoundError(f'mnist_reader not found at {mnist_path}')
        spec = importlib.util.spec_from_file_location('mnist_reader_local', mnist_path)
        mnist_reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mnist_reader)

        kind = 'train' if train else 't10k'
        data_path = os.path.join(fm_root, 'data', 'fashion')
        images, targets = mnist_reader.load_mnist(data_path, kind=kind)

        # images come as (N, 784); reshape to (N, 28, 28)
        images = images.reshape(-1, 28, 28)

        # Keep `.data` and `.targets` attributes to be compatible with store_masked_loaders
        self.data = images
        self.targets = np.array(targets)

    def __getitem__(self, index: int):
        img_arr = self.data[index]
        target = int(self.targets[index])

        # convert to PIL Image (grayscale)
        img = Image.fromarray(img_arr, mode='L')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_not_aug:
            return img, target, not_aug_img
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class SequentialFashionMNIST(ContinualDataset):

    NAME = 'seq-fmnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    # produce 3-channel tensors because backbones expect RGB
    TRANSFORM = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
            self.get_normalization_transform()
        ])

        fm_root = os.path.join(os.path.dirname(__file__), 'fashion_mnist_master')

        train_dataset = MyFashionMNIST(fm_root, train=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
        else:
            test_dataset = MyFashionMNIST(fm_root, train=False, transform=test_transform, return_not_aug=False)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
            self.get_normalization_transform()
        ])

        fm_root = os.path.join(os.path.dirname(__file__), 'fashion_mnist_master')
        train_dataset = MyFashionMNIST(fm_root, train=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose([transforms.ToPILImage(), SequentialFashionMNIST.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialFashionMNIST.N_CLASSES_PER_TASK * SequentialFashionMNIST.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
