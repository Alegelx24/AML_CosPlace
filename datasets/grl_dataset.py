import glob
import random
import logging
import torch
import GradientReversalLayer as GRL
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class GrlDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, datasets_paths, length=1000000):
        """
        datasets_paths is a list containing the folders which contain the N datasets.
        __len__() returns 1000000, and __getitem__(index) returns a random
        image, from dataset index % N, to ensure that each dataset has the 
        same chance of being picked
        """
        super().__init__()
        self.num_classes = len(datasets_paths)
        logging.info(f"GrlDataset has {self.num_classes} classes")
        self.images_paths = []
        for dataset_path in datasets_paths:
            self.images_paths.append(sorted(glob.glob(f"{dataset_root}/{dataset_path}/**/*.jpg", recursive=True)))
            logging.info(f"    Class {dataset_path} has {len(self.images_paths[-1])} images")
            if len(self.images_paths[-1]) == 0:
                raise Exception(f"Class {dataset_path} has 0 images, that's a problem!!!")
        self.transform = GRL.grl_transform
        self.length = length

    def __getitem__(self, index):
        num_class = index % self.num_classes
        images_of_class = self.images_paths[num_class]
        # choose a random one
        image_path = random.choice(images_of_class)
        tensor = self.transform(Image.open(image_path).convert("RGB"))
        return tensor, num_class
    
    def __len__(self):
        return self.length