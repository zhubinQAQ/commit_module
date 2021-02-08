import os
import cv2
import csv
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from head import build_head
from neck import build_neck
from resnet import build_resnet
from test_dataset import TestDataset

model_root = "/home/zhubin/data/mmclassification/work_dirs/resnetv1d152_b32x8_leaf/epoch_100.pth"
csv_root = "/home/zhubin/data/mmclassification/data/leaf/train.csv"
images_root = "/home/zhubin/data/mmclassification/data/leaf/train_images/"
model_dict = dict(
    model_type="ResNetV1d",
    backbone=dict(
        depth=152,
        num_stages=4,
        out_indices=(3,), ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        num_classes=5,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


def get_gt(root):
    gt_dicts = {}
    f = open(root)
    f_csv = csv.reader(f)
    for row in f_csv:
        if len(row[1]) != 1: continue
        gt_dicts[row[0]] = int(row[1])
    return gt_dicts


def get_augmentations():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)  # 86.263

    #     mean = (0.430, 0.496, 0.313)
    #     std = (0.233, 0.235, 0.223) # 83.292

    valid_augmentations = albu.Compose([
        albu.Resize(224, 224),
        albu.Normalize(mean, std, max_pixel_value=255, always_apply=True),
        ToTensorV2(p=1.0)
    ], p=1.0)

    return valid_augmentations


if __name__ == "__main__":
    model = ImageClassifier(model_dict)
    gt_dicts = get_gt(csv_root)
    val_augs = get_augmentations()

    test_ds = TestDataset(images_root, transforms=val_augs)
    test_loader = DataLoader(test_ds, 4, num_workers=4, shuffle=False)

    checkpoint = torch.load(model_root, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    labels = []
    model.cuda()
    right = wrong = 0
    for batch in test_loader:
        x = batch['x'].float().cuda()
        model.eval()
        output = model(x, img_metas=None)
        label = np.array(output).argmax(1)
        for l, i in zip(label, batch['image']):
            if l == gt_dicts[i]:
                right += 1
            else:
                wrong += 1
        labels.extend(label)
        print("right pre: {}/{} {} {:.5f}".format(right, right + wrong, len(labels), float(right) / (right + wrong)))