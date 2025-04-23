import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class lcdDataset(Dataset):
    def __init__(self, data, label, data_list, crop_size=512, have_nd=True):
        self.data = data
        self.label = label
        self.csv = data_list
        self.crop_size = crop_size
        self.have_nd = have_nd

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        Y, X = data.shape

        def rotate(image, angle, value, center=None, scale=1):
            """
            Parameters:
            image:輸入影像，shape 為 (C,H,W)
            angle:旋轉角度
            value:填充旋轉後邊界的數值
            center:旋轉軸心
            scale:是否進行放大
            """
            # 獲取影像尺寸
            (h, w) = image.shape[:2]
            # 若未指定旋轉軸心，則將圖像中心設為旋轉軸心
            if center is None:
                center = (w / 2, h / 2)

            # 執行旋轉
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h), borderValue=value)

            return rotated

        # random crop
        if X != self.crop_size:
            while True:
                origin_x = random.randint(0, X - self.crop_size)
                origin_y = random.randint(0, Y - self.crop_size)

                t_data = data[
                    origin_y : origin_y + self.crop_size,
                    origin_x : origin_x + self.crop_size,
                ]
                t_label = label[
                    origin_y : origin_y + self.crop_size,
                    origin_x : origin_x + self.crop_size,
                ]

                if np.any(t_label > 0):
                    data = t_data
                    label = t_label
                    break

                if not self.have_nd:
                    data = t_data
                    label = t_label
                    break

        # 隨機進行水平翻轉
        rand = random.uniform(0, 1)
        if rand >= 0.5:
            data = cv2.flip(data, 1)
            label = cv2.flip(label, 1)

        # 隨機進行旋轉
        r = random.uniform(-3, 3)
        data = rotate(data, r, -1400)
        label = rotate(label, r, 0)

        data = np.clip(data, -1400, 500)
        label = np.where(label > 0, 1, 0)

        return data, label

    def __len__(self):
        return len(self.csv)
