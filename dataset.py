import glob 
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import ToTensor


DATASET_PATH = 'data/sbd/'
IS_GPU = True
TOTAL_CLASSES = 9

class SegmentationDataset(data.Dataset):
    """
    Data loader for the Segmentation Dataset. If data loading is a bottleneck, 
    you may want to optimize this in for faster training. Possibilities include
    pre-loading all images and annotations into memory before training, so as 
    to limit delays due to disk reads.
    """
    def __init__(self, split="train", data_dir=DATASET_PATH):
        assert(split in ["train", "val", "test"])
        self.img_dir = os.path.join(data_dir, split)
        self.classes = []
        with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
          for l in f:
            self.classes.append(l.rstrip())
        self.n_classes = len(self.classes)
        self.split = split
        self.data = glob.glob(self.img_dir + '/*.jpg') 
        self.data = [os.path.splitext(l)[0] for l in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index] + '.jpg')
        gt = Image.open(self.data[index] + '.png')
        
        img = ToTensor()(img)
        gt = torch.LongTensor(np.asarray(gt)).unsqueeze(0)
        return img, gt

