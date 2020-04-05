# -*- coding: utf-8 -*-
# git clone https://github.com/zhanglideng/At_J_net.git
import sys

sys.path.append('/home/aistudio/external-libraries')
import os
if not os.path.exists('/home/aistudio/.torch/models/vgg16-397923af.pth'):
    os.system('mkdir /home/aistudio/.torch')
    os.system('mkdir /home/aistudio/.torch/models')
    os.system('cp /home/aistudio/work/pre_model/*  /home/aistudio/.torch/models/')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader import AtDataSet
from At_model import *
import time
import xlwt
from utils.ms_ssim import *

LR = 0.0001  # 学习率
EPOCH = 80  # 轮次
BATCH_SIZE = 1  # 批大小
excel_train_line = 1  # train_excel写入的行的下标
excel_val_line = 1  # val_excel写入的行的下标
alpha = 1  # 损失函数的权重
accumulation_steps = 1  # 梯度积累的次数，类似于batch-size=64
# itr_to_lr = 10000 // BATCH_SIZE  # 训练10000次后损失下降50%
itr_to_excel = 4 // BATCH_SIZE  # 训练64次后保存相关数据到excel
loss_num = 3  # 包括参加训练和不参加训练的loss
weight = [1, 1, 1]

# pre_densenet201 = '/home/aistudio/work/pre_model/densenet201.pth'
# pre_vgg16 = '/home/aistudio/work/pre_model/vgg16.pth'
train_haze_path = '/home/aistudio/work/data/cut_ntire_2018/train/'  # 去雾训练集的路径
val_haze_path = '/home/aistudio/work/data/cut_ntire_2018/val/'  # 去雾验证集的路径
gt_path = '/home/aistudio/work/data/cut_ntire_2018/gth/'

save_path = './result_nyu_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
save_model_name = save_path + 'J_model.pt'  # 保存模型的路径
excel_save = save_path + 'result.xls'  # 保存excel的路径
mid_save_ed_path = './J_model/J_model.pt'  # 保存的中间模型，用于下一步训练。

# 初始化excel
f, sheet_train, sheet_val = init_excel()

net = AtJ().cuda()
# print(net)

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 数据转换模式
transform = transforms.Compose([transforms.ToTensor()])
# 读取训练集数据
train_path_list = [train_haze_path, gt_path]
train_data = AtDataSet(transform, train_path_list)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 读取验证集数据
val_path_list = [val_haze_path, gt_path]
val_data = AtDataSet(transform, val_path_list)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.7)

min_loss = 999999999
min_epoch = 0
itr = 0
start_time = time.time()

# 开始训练
print("\nstart to train!")
for epoch in range(EPOCH):
    index = 0
    train_loss = 0
    loss = 0
    loss_excel = [0] * loss_num
    net.train()
    for haze_image, gt_image in train_data_loader:
        index += 1
        itr += 1
        J, A, t, J_reconstruct, haze_reconstruct = net(haze_image)
        loss_image = [J, gt_image]
        loss, temp_loss = loss_function(loss_image, weight)
        train_loss += loss.item()
        loss_excel = [loss_excel[i] + temp_loss[i].item() for i in range(len(loss_excel))]
        loss = loss / accumulation_steps
        loss.backward()
        # 3. update parameters of net
        if ((index + 1) % accumulation_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        if np.mod(index, itr_to_excel) == 0:
            loss_excel = [loss_excel[i] / itr_to_excel for i in range(len(loss_excel))]
            print('epoch %d, %03d/%d' % (epoch + 1, index, len(train_data_loader)))
            print('L2=%.5f\n' 'SSIM=%.5f\n' 'VGG=%.5f\n' % (loss_excel[0], loss_excel[1], loss_excel[2]))
            print_time(start_time, index, EPOCH, len(train_data_loader), epoch)
            excel_train_line = write_excel(sheet=sheet_train,
                                           data_type='train',
                                           line=excel_train_line,
                                           epoch=epoch,
                                           itr=itr,
                                           loss=loss_excel,
                                           weight=weight)
            f.save(excel_save)
            loss_excel = [0] * loss_num
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    loss_excel = [0] * loss_num
    with torch.no_grad():
        net.eval()
        for haze_image, gt_image in val_data_loader:
            J, A, t, J_reconstruct, haze_reconstruct = net(haze_image)
            loss_image = [J, gt_image]
            loss, temp_loss = loss_function(loss_image, weight)
            loss_excel = [loss_excel[i] + temp_loss[i].item() for i in range(len(loss_excel))]
    train_loss = train_loss / len(train_data_loader)
    val_loss = sum(loss_excel)
    loss_excel = [loss_excel[i] / len(val_data_loader) for i in range(len(loss_excel))]
    print('L2=%.5f\n' 'SSIM=%.5f\n' 'VGG=%.5f\n' % (loss_excel[0], loss_excel[1], loss_excel[2]))
    excel_val_line = write_excel(sheet=sheet_val,
                                 data_type='val',
                                 line=excel_val_line,
                                 epoch=epoch,
                                 itr=False,
                                 loss=[loss_excel, val_loss, train_loss],
                                 weight=False)
    f.save(excel_save)
    if val_loss < min_loss:
        min_loss = val_loss
        min_epoch = epoch
        torch.save(net, save_model_name)
        print('saving the epoch %d model with %.5f' % (epoch + 1, min_loss))
print('Train is Done!')
