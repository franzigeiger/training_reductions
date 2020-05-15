import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data


class DataSubSet(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, imagenetDataset, fraction, transform):
        'Initialization'
        # classes (list): List of the class names.
        # class_to_idx (dict): Dict with items (class_name, class_index).
        # imgs (list): List of (image path, class_index) tuples
        self.classes = imagenetDataset.classes
        self.class_to_idx = imagenetDataset.class_to_idx
        self.transform = transform
        imgs = imagenetDataset.imgs
        items = len(imgs)
        new_imgs = []
        if fraction > 1:
            new_items = fraction
        else:
            new_items = int(items * fraction)
        mask = np.concatenate([np.ones(new_items), np.zeros(items - new_items)])
        np.random.shuffle(mask)
        for i in range(items):
            if mask[i] == 1:
                new_imgs.append(imgs[i])
        self.imgs = new_imgs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        (path, lab) = self.imgs[index]

        # Load data and get label
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        X = self.transform(img)
        # y = self.classes[lab]
        y = torch.Tensor(lab)
        return X, lab


normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
data_path = '/braintree/data2/active/common/imagenet_raw/' if 'IMAGENET' not in os.environ else os.environ['IMAGENET']


def get_dataloader(image_load=100, batch_size=256, workers=20):
    images = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'train'),
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))
    dataset = DataSubSet(images, image_load, torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
    ]))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=workers,
                                              pin_memory=True)
    return data_loader
