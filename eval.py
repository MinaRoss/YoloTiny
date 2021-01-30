import time
import cv2
import torch
from yolo_tiny.utils import resize, nms, return2RealPosition
from yolo_tiny.detector import Detector
from utils import init_settings
import argparse
import datetime


def Eval(path, net_path=None):
    img, expand, scale = resize(cv2.imread(path))
    data = torch.tensor(img.transpose([2, 0, 1]) / 255 - 0.5,
                        dtype=torch.float32).unsqueeze(dim=0)
    setting = init_settings()
    if net_path:
        setting['net_path'] = net_path
        print('[{}]指定的网络文件路径已起效'.format(datetime.datetime.now()))
    yolo = Detector(setting['net_path'])

    startTime = time.time()
    boxes = yolo.detect(data, 0.5, setting['anchors'])  # Box <- [batch,confi,cx,cy,w,h,cls]
    boxes = boxes.cpu()
    stopTime = time.time()
    print('* ------------------------------------------ *')
    print('* PROCESSING TIME COST : {}'.format(stopTime - startTime))

    if boxes.size()[0] == 0:
        print('* NO THINGS CAUGHT')
        return boxes.numpy()

    frame = nms(boxes, 0.5, True).cpu().detach().numpy()  # box_idx, [N, IOU, CX, CY, W, H, CLS]
    frame[:, 2:6] = return2RealPosition(frame[:, 2:6], expand, scale)
    print('* NUM OF BOXES : {} / {}'.format(frame.shape[0], boxes.size()[0]))
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--path', nargs='?', type=str, help='文件路径', required=True)
    parser.add_argument('--net_path', nargs='?', default=None, type=str, help='网络路径', required=False)
    parser.set_defaults(tboard=False)
    args = parser.parse_args()
    boxes = Eval(args.path, args.net_path)
    img = cv2.imread(args.path)
    for n, iou, x, y, w, h, *cls in boxes:
        print("* Object IOU : {}".format(iou))
        print("* Cls : {}".format(cls))
        x, y, w, h = list(map(int, [x, y, w, h]))
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.imshow('detection', img)
    cv2.waitKey(0)
