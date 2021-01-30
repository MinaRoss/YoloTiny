import torch
from .sampler import Sampling
from torch.utils.data import DataLoader
import os
from .yolo_tiny import YoloTiny
import torch.optim as opt
import torch.nn as nn
import datetime


def Trainer(data_dir, yolo_net_file_path, launch_mode, epochs, batch_size, anchors, areas, is_new=False):
    print("[{}][{}]网络训练程序启动中...".format(datetime.datetime.now(), launch_mode))
    print("**************************************************************")
    yolo_net_file_dir = "/".join(yolo_net_file_path.split("/")[:-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Sampling(data_dir, anchors, areas)
    train_data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    if os.path.exists(yolo_net_file_path) and not is_new:
        yolo = torch.load(yolo_net_file_path, map_location=device.type)
        print('* 已加载存在的网络文件 {}'.format(yolo_net_file_path))
    else:
        yolo = YoloTiny().to(device)

    yolo.train()
    optimizer = opt.Adam(yolo.parameters())
    loss_mse = nn.MSELoss()

    ave_loss = 1000.
    saved_epoch = 0
    print('* 使用中的设备 : {} | {}'.format(device.type, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'))
    print('* 训练轮次 : {}'.format(epochs))
    print('* 批次大小 : {}'.format(batch_size))
    print('* 优化器 : Adam')
    print("**************************************************************")
    for epoch in range(epochs):
        sum_loss = 0.
        for idx, (data, label26, label13) in enumerate(train_data):
            label26, label13, data = label26.to(device), label13.to(device), data.to(device)
            out13, out26 = yolo(data)

            loss26 = calc_loss(out26, label26, 0.7, loss_mse)
            loss13 = calc_loss(out13, label13, 0.7, loss_mse)
            loss = loss26 + loss13
            sum_loss += loss
            print('\r* [轮次 : {}][{}/{}] 损失 : {}'.format(epoch, idx + 1, len(train_data), loss), end='')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (sum_loss / len(train_data)) < ave_loss:
            ave_loss = sum_loss / len(train_data)
            saved_epoch = epoch

            if not os.path.exists(yolo_net_file_dir):
                os.makedirs(yolo_net_file_dir)

            if int("".join(torch.__version__.split('.'))) < 160:
                torch.save(yolo, yolo_net_file_path)
            else:
                torch.save(yolo, yolo_net_file_path, _use_new_zipfile_serialization=False)
            print('\r* [轮次 : {}]平均损失 : {} | 网络文件已保存'.format(epoch, sum_loss / len(train_data)))
        else:
            print('\r* [轮次 : {}]平均损失 : {} | 保存文件为轮次{}网络文件'.format(epoch, sum_loss / len(train_data), saved_epoch))


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
