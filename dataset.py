import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import image

class RGBDataset(Dataset):
    def __init__(self, dataset_dir, has_gt, is_train=False):
        """
        In:
            dataset_dir: string, train_dir, val_dir, and test_dir in segmentation.py.
                         Be careful the images are stored in the subfolders under these directories.
            has_gt: bool, indicating if the dataset has ground truth masks.
            is_train: bool, whether to apply data augmentations.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        """
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
        self.is_train = is_train
        # TODO: transform to be applied on a sample.
        #  For this homework, compose transforms.ToTensor() and transforms.Normalize() for RGB image should be enough.
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_rgb, std=std_rgb)])
        # TODO: number of samples in the dataset.
        #  You'd better not hard code the number,
        #  because this class is used to create train, validation and test dataset (which have different sizes).
        rgb_dir = os.path.join(dataset_dir, "rgb")
        self.dataset_length = len(os.listdir(rgb_dir))
        

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        # TODO: read RGB image and ground truth mask, apply the transformation, and pair them as a sample.
        rgb_dir = os.path.join(self.dataset_dir, "rgb")
        rgb_filename = f"{idx}_rgb.png"
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        rgb_img = image.read_rgb(rgb_path)
        rgb_pil = Image.fromarray(rgb_img)
        
        if self.has_gt:
            gt_dir = os.path.join(self.dataset_dir, "gt")
            gt_filename = f"{idx}_gt.png"
            gt_path = os.path.join(gt_dir, gt_filename)
            gt_mask = image.read_mask(gt_path)
            gt_pil = Image.fromarray(gt_mask)

            if self.is_train:
                if random.random() > 0.5:
                    rgb_pil = TF.hflip(rgb_pil)
                    gt_pil = TF.hflip(gt_pil)
                
                angle = random.uniform(-15, 15)
                rgb_pil = TF.rotate(rgb_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
                gt_pil = TF.rotate(gt_pil, angle, interpolation=TF.InterpolationMode.NEAREST)
                
                rgb_pil = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(rgb_pil)
                
            sample = {'input': self.transform(rgb_pil), 'target': torch.LongTensor(np.array(gt_pil))}
        else:
            sample = {'input': self.transform(rgb_pil)}
            
        return sample