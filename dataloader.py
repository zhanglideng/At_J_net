from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio


# nyu/test/1318_a=0.55_b=1.21.png
class AtDataSet(Dataset):
    def __init__(self, transform1, path=None):
        # print(path)
        self.transform1 = transform1
        self.haze_path, self.gt_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.gt_data_list = os.listdir(self.gt_path)

        self.haze_data_list.sort(key=lambda x: float(x[:-4]))
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))

        self.length = len(os.listdir(self.haze_path))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
            需要传递的信息有：
            有雾图像
            无雾图像
            (深度图)
            (雾度)
            (大气光)
            624, 464
        """

        haze_image_name = self.haze_data_list[idx]
        haze_image = cv2.imread(self.haze_path + haze_image_name)
        gt_image = cv2.imread(self.gt_path + haze_image_name)
        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)
        return haze_image.cuda(), gt_image.cuda()

# if __name__ == '__main__':
