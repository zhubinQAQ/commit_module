import cv2
import os
import pandas as pd

import torch.nn as nn
from torch.utils.data import Dataset

from head import build_head
from neck import build_neck
from resnet import build_resnet


class TestDataset(Dataset):
    def __init__(self, images_root, transforms=None):
        test_df = pd.DataFrame()
        test_df['image_id'] = list(os.listdir(images_root))
        self.images_root = images_root
        self.test_df = test_df
        self.transforms = transforms

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        image_path = os.path.join(self.images_root, self.test_df.iloc[index].image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (self.transforms):
            image = self.transforms(image=image)["image"]

        return {
            "x": image,
            "image": self.test_df.iloc[index].image_id,
        }


class ImageClassifier(nn.Module):
    def __init__(self, model_dict):
        super(ImageClassifier, self).__init__()
        model_type = model_dict['model_type']
        self.backbone = build_resnet(model_dict['backbone'], model_type)
        self.neck = build_neck()
        self.head = build_head(model_dict['head'])

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)