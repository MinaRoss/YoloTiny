from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import math


class Sampling(Dataset):
    def __init__(self, data_dir, anchors, areas):
        image_dir = os.path.join(data_dir, r'image')
        label_path = os.path.join(data_dir, r'label.txt')
        self.labels = []
        self.image_paths = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                file, *label = line.split()
                self.image_paths.append(os.path.join(image_dir, file))
                self.labels.append(list(map(lambda x: x.split('.'), label)))
        self.anchors = anchors
        self.areas = areas

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        labels = {}
        img_data = cv2.imread(self.image_paths[index]).transpose([2, 0, 1]).astype(np.float32) / 255 - 0.5

        for featureSize, sub_anchors in self.anchors.items():  # per feature size
            labels[featureSize] = np.zeros(shape=(featureSize, featureSize, 3, 5), dtype=np.float32)
            for box in self.labels[index]:
                cx, cy, w, h, *cls = list(map(int, box))  # centre x & y, w & h
                cx_offset, cx_idx = math.modf(cx * featureSize / 416)
                cy_offset, cy_idx = math.modf(cy * featureSize / 416)
                for i, anchor in enumerate(sub_anchors):  # per box in one feature size
                    parea = self.areas[featureSize][i]
                    pw, ph = w / anchor[0], h / anchor[1]
                    barea = w * h
                    iou = min(parea, barea) / max(parea, barea)
                    labels[featureSize][int(cy_idx), int(cx_idx), i] = np.array([iou, cx_offset, cy_offset, np.log(pw), np.log(ph), *cls])

        return img_data, labels[26], labels[13]
