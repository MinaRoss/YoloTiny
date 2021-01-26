from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import math
import json


class Sampling(Dataset):
    def __init__(self, data_dir):
        files = os.listdir(data_dir)
        self.data_paths = [os.path.join(data_dir, file) for file in files]
        self.labels = [file.split('.')[:-2] for file in files]  # drop last two strings (pic order & file type)
        try:
            with open(r'./.config') as f:
                self.settings = json.load(f)
                self.anchors = self.settings['anchors']
                self.areas = self.settings['areas']
        except Exception as e:
            raise e

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        labels = {}
        img_data = cv2.imread(self.data_paths[index]).transpose([2, 0, 1]).astype(np.float32) / 255 - 0.5
        is_positive = bool(int(self.labels[index][0]))
        box = np.array([float(x) for x in self.labels[index][1:5]])
        cls = np.array([float(x) for x in self.labels[index][5:]], dtype=np.float32)

        for featureSize, sub_anchors in self.anchors.items():  # per feature size
            labels[featureSize] = np.zeros(shape=(featureSize, featureSize, 3, 5), dtype=np.float32)

            if is_positive:
                cx, cy, w, h = box  # centre x & y, w & h
                cx_offset, cx_idx = math.modf(cx * featureSize / 416)
                cy_offset, cy_idx = math.modf(cy * featureSize / 416)
                for i, anchor in enumerate(sub_anchors):  # per box in one feature size
                    parea = self.areas[featureSize][i]
                    pw, ph = w / anchor[0], h / anchor[1]
                    barea = w * h
                    iou = min(parea, barea) / max(parea, barea)
                    labels[featureSize][int(cy_idx), int(cx_idx), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(pw), np.log(ph)])

        return img_data, labels[26], labels[13], cls
