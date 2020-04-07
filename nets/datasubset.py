import numpy as np
import torch
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
        new_items = int(items * fraction)
        new_imgs = []
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
