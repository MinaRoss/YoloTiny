import torch
import cv2


def iou(box, boxes, wholeCover=False):
    '''
    Calculate the cover rate between two boxes
    :param box: the box has the max confidence, shape like [4], type like 'Ndarray'
    :param boxes: the boxes will be calculated, shape like [N,4], type like 'Ndarray'
    :return: cover rate, type like 'Float'
    '''
    x1, y1, x2, y2 = box[2] - box[4] / 2, \
                     box[3] - box[5] / 2, \
                     box[2] + box[4] / 2, \
                     box[3] + box[5] / 2
    x1s, y1s, x2s, y2s = boxes[:, 2] - boxes[:, 4] / 2, \
                         boxes[:, 3] - boxes[:, 5] / 2, \
                         boxes[:, 2] + boxes[:, 4] / 2, \
                         boxes[:, 3] + boxes[:, 5] / 2
    area = torch.abs(x2 - x1) * torch.abs(y2 - y1)
    areas = torch.abs(x2s - x1s) * torch.abs(y2s - y1s)
    x1m, y1m, x2m, y2m = torch.max(x1, x1s), torch.max(y1, y1s), torch.min(x2, x2s), torch.min(y2, y2s)
    cover = torch.max(x2m - x1m, torch.tensor(0, dtype=torch.float32)) * \
            torch.max(y2m - y1m, torch.tensor(0, dtype=torch.float32))
    rate = torch.div(cover, area + areas - cover) if not wholeCover \
        else torch.div(cover, torch.min(area, areas))

    return rate


def nms(boxes, coverMin, wholeCover=False):
    '''
    Order the boxes in confidence and calculate the cover rate between boxes
    :param boxes: frame will be calculated, shape like [N,5], type like 'Ndarray'
    :param coverMin: the minimum cover rate between boxes belonging to the same target, type like 'Float'
    :param wholeCover: set this parameter as 'True' for checking whole covering, default as 'False'
    :return: boxes belonging to different targets, shape like [N,5], type like 'Ndarray'
    '''
    if boxes.size()[0] <= 1:
        return boxes
    boxes = boxes[(-boxes[:, 1]).argsort()]
    keep = []
    while boxes.size()[0] > 1:
        box = boxes[0]
        boxes = boxes[1:]
        keep.append(box)
        boxes = boxes[iou(box, boxes, wholeCover) < coverMin]  # calculate the cover rate between boxes
    if boxes.size()[0] == 1:
        keep.append(boxes[0])

    return torch.stack(keep)


def resize(img, size=(416, 416)):
    h, w = img.shape[:2]
    if 1.0 > h / w:
        resize_h = (1.0 * w - h) * 0.5
        expand = (int(resize_h), int(resize_h), 0, 0)
    elif 1.0 < h / w:
        resize_w = (h / 1.0 - w) * 0.5
        expand = (0, 0, int(resize_w), int(resize_w))
    else:
        expand = (0, 0, 0, 0)

    expand_img = cv2.copyMakeBorder(img, *expand, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ratio = (416 / expand_img.shape[0] + 416 / expand_img.shape[1]) * 0.5
    return cv2.resize(expand_img, size), expand, ratio
