"""Data utility functions."""
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
import random

import _pickle as pickle

# pylint: disable=C0326
SEG_LABELS_LIST = [
    {"id": -1, "name": "void",       "rgb_values": [0,   0,    0]},
    {"id": 0,  "name": "building",   "rgb_values": [128, 0,    0]},
    {"id": 1,  "name": "grass",      "rgb_values": [0,   128,  0]},
    {"id": 2,  "name": "tree",       "rgb_values": [128, 128,  0]},
    {"id": 3,  "name": "cow",        "rgb_values": [0,   0,    128]},
    {"id": 4,  "name": "horse",      "rgb_values": [128, 0,    128]},
    {"id": 5,  "name": "sheep",      "rgb_values": [0,   128,  128]},
    {"id": 6,  "name": "sky",        "rgb_values": [128, 128,  128]},
    {"id": 7,  "name": "mountain",   "rgb_values": [64,  0,    0]},
    {"id": 8,  "name": "airplane",   "rgb_values": [192, 0,    0]},
    {"id": 9,  "name": "water",      "rgb_values": [64,  128,  0]},
    {"id": 10, "name": "face",       "rgb_values": [192, 128,  0]},
    {"id": 11, "name": "car",        "rgb_values": [64,  0,    128]},
    {"id": 12, "name": "bicycle",    "rgb_values": [192, 0,    128]},
    {"id": 13, "name": "flower",     "rgb_values": [64,  128,  128]},
    {"id": 14, "name": "sign",       "rgb_values": [192, 128,  128]},
    {"id": 15, "name": "bird",       "rgb_values": [0,   64,   0]},
    {"id": 16, "name": "book",       "rgb_values": [128, 64,   0]},
    {"id": 17, "name": "chair",      "rgb_values": [0,   192,  0]},
    {"id": 18, "name": "road",       "rgb_values": [128, 64,   128]},
    {"id": 19, "name": "cat",        "rgb_values": [0,   192,  128]},
    {"id": 20, "name": "dog",        "rgb_values": [128, 192,  128]},
    {"id": 21, "name": "body",       "rgb_values": [64,  64,   0]},
    {"id": 22, "name": "boat",       "rgb_values": [192, 64,   0]}]


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


def compute_class_weights(dataset, num_classes=23):
    """Compute class weights based on class frequencies in the dataset."""
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    for _, target in dataset:
        # Ignore void class (-1)
        mask = target >= 0
        target_valid = target[mask]
        
        # Count class occurrences
        for i in range(num_classes):
            class_counts[i] += (target_valid == i).sum().item()
        total_pixels += mask.sum().item()
    
    # Calculate weights as inverse of frequency
    class_weights = total_pixels / (class_counts * num_classes)
    # Handle zero counts
    class_weights[class_counts == 0] = 0
    return class_weights

class SegmentationTransform:
    def __init__(self, is_training=True, crop_size=240):
        self.is_training = is_training
        self.crop_size = crop_size
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
        
    def __call__(self, img, target):
        # Convert to PIL for transforms
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        if isinstance(target, torch.Tensor):
            target = Image.fromarray(target.numpy().astype(np.uint8))
            
        # Training augmentations
        if self.is_training:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                target = TF.hflip(target)
            
            # Random rotation
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                img = TF.rotate(img, angle, fill=0)
                target = TF.rotate(target, angle, fill=255)  # Use 255 as ignore index
            
            # Random color augmentation
            if random.random() > 0.3:
                img = self.color_jitter(img)
            
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(self.crop_size, self.crop_size))
            img = TF.crop(img, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
        else:
            # Center crop for validation/test
            img = TF.center_crop(img, self.crop_size)
            target = TF.center_crop(target, self.crop_size)
        
        # Convert back to tensor
        img = TF.to_tensor(img)
        target = torch.from_numpy(np.array(target))
        
        # Map void class (-1) to ignore index (255)
        target[target == -1] = 255
        
        return img, target

class SegmentationData(data.Dataset):
    def __init__(self, image_paths_file, is_training=False):
        self.root_dir_name = os.path.dirname(image_paths_file)
        self.is_training = is_training
        self.transform = SegmentationTransform(is_training=is_training)

        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        img_id = self.image_names[index].replace('.bmp', '')

        # Load image
        img = Image.open(os.path.join(self.root_dir_name,
                                    'images',
                                    img_id + '.bmp')).convert('RGB')
        
        # Load target
        target = Image.open(os.path.join(self.root_dir_name,
                                       'targets',
                                       img_id + '_GT.bmp'))
        target = np.array(target, dtype=np.int64)
        
        # Convert RGB target to class indices
        target_labels = target[..., 0].copy()
        for label in SEG_LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis=2)
            target_labels[mask] = label['id']
        
        target_labels = torch.from_numpy(target_labels)
        
        # Apply transforms
        img = transforms.ToTensor()(img)
        img, target_labels = self.transform(img, target_labels)
        
        return img, target_labels
