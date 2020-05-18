from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio
import torch


# nyu/test/1318_a=0.55_b=1.21.png
class AtJDataSet(Dataset):
    def __init__(self, transform1, path=None, flag='train'):
        # print(path)
        self.flag = flag
        self.transform1 = transform1
        self.haze_path, self.gt_path, self.t_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.gt_data_list = os.listdir(self.gt_path)
        self.t_data_list = os.listdir(self.t_path)

        self.haze_data_list.sort(key=lambda x: int(x[:-18]))

        self.length = len(os.listdir(self.haze_path))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        haze_image_name = self.haze_data_list[idx]

        A_gth = np.ones((608, 448, 3), dtype=np.float32)
        A = float(haze_image_name[-15:-11])
        A_gth = A_gth * A

        t_gth = np.load(self.t_path + haze_image_name[:-4] + '.npy')
        t_gth = np.expand_dims(t_gth, axis=2)
        t_gth = t_gth.astype(np.float32)

        haze_image = cv2.imread(self.haze_path + haze_image_name)
        gt_image = cv2.imread(self.gt_path + haze_image_name[:-18] + '.PNG')
        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)
            A_gth = self.transform1(A_gth)
            t_gth = self.transform1(t_gth)

        print(t_gth[0][0][0])
        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        A_gth = A_gth.cuda()
        t_gth = t_gth.cuda()
        if self.flag == 'train':
            return haze_image, gt_image, A_gth, t_gth
        elif self.flag == 'test':
            return haze_image_name, haze_image, gt_image, A_gth, t_gth

        # if __name__ == '__main__':
