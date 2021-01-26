import torch
from .sampler import Sampling
from torch.utils.data import DataLoader
import os
from .yolo_tiny import YoloTiny
import torch.optim as opt
import torch.nn as nn
import datetime


def Trainer(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Sampling(data_dir)
    train_data = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    if os.path.exists(yolo_net_file_path):
        yolo = torch.load(yolo_net_file_path, map_location=device.type)
        print('* LOADED EXISTED NET FILE')
    else:
        yolo = YoloTiny().to(device)

    yolo.train()
    optimizer = opt.Adam(yolo.parameters())
    loss_mse = nn.MSELoss()

    epoch = 0
    ave_loss = 1000.
    saved_epoch = 0
    print('* {}'.format(datetime.datetime.now()))
    print('* DEVICE : {} | {}'.format(device.type,
                                      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'))
    while True:
        sum_loss = 0.
        epoch += 1
        for idx, (data, label26, label13, cls) in enumerate(train_data):
            cls, label26, label13, data = cls.to(device), label26.to(device), label13.to(device), data.to(device)
            out13, out26, out_cls = yolo(data)

            loss26 = calc_loss(out26, label26, 0.7, loss_mse)
            loss13 = calc_loss(out13, label13, 0.7, loss_mse)
            loss_cls = loss_mse(out_cls, cls)
            loss = loss26 + loss13 + loss_cls
            sum_loss += loss
            print('\r* EPOCH : {} * {}/{} * LOSS : {}'.format(epoch, idx + 1, len(train_data), loss), end='')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (sum_loss / len(train_data)) < ave_loss:
            ave_loss = sum_loss / len(train_data)
            saved_epoch = epoch
            torch.save(yolo, yolo_net_file_path, _use_new_zipfile_serialization=False)
            print('\r* EPOCH : {} * AVE_LOSS : {} * SAVED'.format(epoch, sum_loss / len(train_data)))
        else:
            print('\r* EPOCH : {} * AVE_LOSS : {} * USING AT {}'.format(epoch, sum_loss / len(train_data), saved_epoch))


def calc_loss(output, label, alpha, loss_fn):
    output = output.permute(0, 2, 3, 1)  # Axis transpose, [N,C,H,W] -> [N,H,W,C]
    output = output.reshape(output.size()[0], output.size()[1], output.size()[2], 3,
                            -1)  # At the last dimension, C -> 3 * per box, N batch, [H, W] idx

    maskWithObj = (label[..., 0] > 0.)
    maskWithoutObj = (label[..., 0] == 0.)
    loss_box_WithObj = loss_fn(output[maskWithObj],
                               label[maskWithObj])  # For positive samples, it will train IOU and COORDINATE
    lossWithoutObj = loss_fn(output[maskWithoutObj][:, 0],
                             label[maskWithoutObj][:, 0])  # For negative samples, it only need to train IOU, but not COORDINATE
    return loss_box_WithObj * alpha + lossWithoutObj * (1 - alpha)


if __name__ == '__main__':
    Trainer(dataset_dir)
