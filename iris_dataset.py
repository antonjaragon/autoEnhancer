import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image, ImageCms
from matplotlib import pyplot as plt
import cv2
import random
import torchvision
import numpy as np
from joint_transforms import crop, scale, flip, rotate
from torchvision import transforms

def convert_from_image_to_cv2(img):
    return np.array(img)

def convert_from_BGR_to_RGB(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

def convert_from_image_to_hsv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

def convert_from_image_to_lab(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)


class IrisImageFolderTrain(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.images = os.listdir(os.path.join(root, 'samples'))
        self.labels = os.listdir(os.path.join(root, 'labels'))
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'samples', self.images[index])).convert('RGB')
        target = Image.open(os.path.join(self.root, 'labels', self.labels[index])).convert('RGB')
        # print(self.imgs[index], '--', self.labels[index])

        img_list = []
        gt_list = []
        img_list.append(img)
        img_list.append(target)
        gt_list.append(img.convert('L'))
        gt_list.append(img.convert('L'))
        if self.joint_transform is not None:
            img_list, gt_list = self.joint_transform(img_list, gt_list)

        img = img_list[0]
        # hsv = img_list[0].convert('HSV')
        hsv = convert_from_image_to_hsv(img_list[0])
        # lab = img_list[0].convert('HSV')
        lab = convert_from_image_to_lab(img_list[0])
        lab_target = convert_from_image_to_lab(img_list[1])
        target = convert_from_image_to_cv2(img_list[1])
        if self.transform is not None:
            img = self.transform(img)
            hsv = self.transform(hsv)
            lab = self.transform(lab)
            target = self.transform(target)
            lab_target = self.transform(lab_target)

        return img, hsv, lab, target, lab_target

    def __len__(self):
        return len(self.images)
    
            
        




        