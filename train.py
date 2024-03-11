import argparse
import configparser
import datetime
import os
import time
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
"""
这段代码导入了一些必要的库和模块，下面我将逐一介绍每个包的作用：
1. argparse：用于解析命令行参数，可以方便地从命令行中获取用户输入的参数。
2. os：提供了许多与操作系统交互的函数，用于处理文件和目录。
3. collections 中的 OrderedDict：是一个有序字典，用于创建有序的键值对。
4. glob：根据指定的路径模式匹配文件，可以用来获取符合条件的文件列表。
5. pandas：用于数据分析和处理，提供了高性能、易用的数据结构和数据分析工具。
6. torch：PyTorch深度学习框架的核心库，提供了张量操作、自动求导等功能。
7. torch.backends.cudnn：用于设置一些与CUDA加速相关的选项，提供了对cuDNN库的接口。
8. torch.nn：PyTorch中用于定义神经网络层的模块，包括各种不同类型的层和损失函数。
9. torch.optim：提供了各种优化器的实现，用于更新模型的参数。
10. yaml：用于读取和写入YAML格式的配置文件。
11. albumentations：一个图像增强库，提供了各种图像增强的方法，如旋转、缩放、裁剪等。
12. transforms：albumentations中的模块，提供了各种图像增强的操作。
13. Compose：albumentations中的模块，用于将多个图像增强操作组合在一起。
14. train_test_split：用于将数据集划分为训练集和验证集的模块。
15. torch.optim.lr_scheduler：PyTorch中的学习率调度器，用于动态调整学习率。
16. tqdm：一个Python进度条库，用于在循环中显示进度条。
17. archs：自定义的模型架构，用于定义神经网络模型。
18. losses：自定义的损失函数，用于定义模型的损失计算方法。
19. dataset：自定义的数据集类，用于加载和处理数据。
20. metrics：自定义的评估指标，用于评估模型性能。
21. utils：自定义的工具函数，用于辅助操作和处理。
"""



"""
ARCH_NAMES = archs.__all__：从 archs 模块中导入所有可用的模型架构的名称，并将它们存储在 ARCH_NAMES 列表中。
LOSS_NAMES = losses.__all__：从 losses 模块中导入所有可用的损失函数的名称，并将它们存储在 LOSS_NAMES 列表中。
LOSS_NAMES.append('BCEWithLogitsLoss')：将一个名为 'BCEWithLogitsLoss' 的损失函数名称添加到 LOSS_NAMES 列表中。
"""
ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

"""
命令行参数： 指定参数：
--dataset dsb2018_96 
--arch NestedUNet
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
这段代码是用来解析命令行参数的函数。下面是每一句代码的作用：
1. `parser = argparse.ArgumentParser()`：创建一个ArgumentParser对象，用于解析命令行参数。
2. `parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')`：添加一个名为'name'的参数，用于指定模型的名称，默认值为None，帮助信息为'model name: (default: arch+timestamp)'。
3. `parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')`：添加一个名为'epochs'的参数，用于指定总共运行的训练轮数，默认值为10，类型为整数，帮助信息为'number of total epochs to run'。
4. `parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 16)')`：添加一个名为'batch_size'的参数，用于指定每个mini-batch的样本数量，默认值为8，类型为整数，帮助信息为'mini-batch size (default: 16)'。
5. `parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet', choices=ARCH_NAMES, help='model architecture: ' + ' | '.join(ARCH_NAMES) + ' (default: NestedUNet)')`：添加一个名为'arch'或'a'的参数，用于指定模型的架构，默认值为'NestedUNet'，可选值为ARCH_NAMES中的架构名称，帮助信息为'model architecture: ' + ' | '.join(ARCH_NAMES) + ' (default: NestedUNet)'。
6. `parser.add_argument('--deep_supervision', default=False, type=str2bool)`：添加一个名为'deep_supervision'的参数，用于指定是否使用深度监督，默认值为False。
7. `parser.add_argument('--input_channels', default=3, type=int, help='input channels')`：添加一个名为'input_channels'的参数，用于指定输入图像的通道数，默认值为3，帮助信息为'input channels'。
8. `parser.add_argument('--num_classes', default=1, type=int, help='number of classes')`：添加一个名为'num_classes'的参数，用于指定分类的类别数量，默认值为1，帮助信息为'number of classes'。
9. `parser.add_argument('--input_w', default=96, type=int, help='image width')`：添加一个名为'input_w'的参数，用于指定输入图像的宽度，默认值为96，帮助信息为'image width'。
10. `parser.add_argument('--input_h', default=96, type=int, help='image height')`：添加一个名为'input_h'的参数，用于指定输入图像的高度，默认值为96，帮助信息为'image height'。
11. `parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES, help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceLoss)')`：添加一个名为'loss'的参数，用于指定损失函数，默认值为'BCEDiceLoss'，可选值为LOSS_NAMES中的损失函数名称，帮助信息为'loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceLoss)'。
12. `parser.add_argument('--dataset', default='dsb2018_96', help='dataset name')`：添加一个名为'dataset'的参数，用于指定数据集的名称，默认值为'dsb2018_96'，帮助信息为'dataset name'。
13. `parser.add_argument('--img_ext', default='.png', help='image file extension')`：添加一个名为'img_ext'的参数，用于指定图像文件的扩展名，默认值为'.png'，帮助信息为'image file extension'。
14. `parser.add_argument('--mask_ext', default='.png', help='mask file extension')`：添加一个名为'mask_ext'的参数，用于指定掩码文件的扩展名，默认值为'.png'，帮助信息为'mask file extension'。
15. `parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'], help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')`：添加一个名为'optimizer'的参数，用于指定优化器，默认值为'SGD'，可选值为['Adam', 'SGD']，帮助信息为'loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)'。
16. `parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')`：添加一个名为'lr'或'learning_rate'的参数，用于指定初始学习率，默认值为1e-3，类型为浮点数，帮助信息为'initial learning rate'。
17. `parser.add_argument('--momentum', default=0.9, type=float, help='momentum')`：添加一个名为'momentum'的参数，用于指定动量，默认值为0.9。
18. `parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')`：添加一个名为'weight_decay'的参数，用于指定权重衰减，默认值为1e-4。
19. `parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')`：添加一个名为'nesterov'的参数，用于指定是否使用Nesterov动量，默认值为False。
20. `parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])`：添加一个名为'scheduler'的参数，用于指定学习率调度器，默认值为'CosineAnnealingLR'，可选值为['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR']。
21. `parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')`：添加一个名为'min_lr'的参数，用于指定最小学习率，默认值为1e-5。
22. `parser.add_argument('--factor', default=0.1, type=float)`：添加一个名为'factor'的参数，用于指定学习率调度器中的因子，默认值为0.1。
23. `parser.add_argument('--patience', default=2, type=int)`：添加一个名为'patience'的参数，用于指定学习率调度器中的耐心值，默认值为2。
24. `parser.add_argument('--milestones', default='1,2', type=str)`：添加一个名为'milestones'的参数，用于指定学习率调度器中的里程碑，默认值为'1,2'。
25. `parser.add_argument('--gamma', default=2 / 3, type=float)`：添加一个名为'gamma'的参数，用于指定学习率调度器中的γ值，默认值为2/3。
26. `parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')`：添加一个名为'early_stopping'的参数，用于指定早停的轮数，默认值为-1，帮助信息为'early stopping (default: -1)'。
27. `parser.add_argument('--num_workers', default=0, type=int)`：添加一个名为'num_workers'的参数，用于指定数据加载时的并行工作进程数，默认值为0。
28. `config = parser.parse_args()`：解析命令行参数，并将结果存储在config变量中。
29. `return config`：返回解析后的命令行参数配置。
"""
def original_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

# 使用ini配置文件添入默认值
def parse_args():
    parser = argparse.ArgumentParser()

    config = configparser.ConfigParser()
    config.read('resources/defaultValue.ini')

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=config.getint('training', 'epochs'), type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=config.getint('training', 'batch_size'), type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default=config.get('training', 'arch'),
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=config.getint('training', 'input_channels'), type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=config.getint('training', 'num_classes'), type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=config.getint('training', 'input_w'), type=int,
                        help='image width')
    parser.add_argument('--input_h', default=config.getint('training', 'input_h'), type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default=config.get('training', 'loss'),
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default=config.get('data', 'dataset'),
                        help='dataset name')
    parser.add_argument('--img_ext', default=config.get('data', 'img_ext'),
                        help='image file extension')
    parser.add_argument('--mask_ext', default=config.get('data', 'mask_ext'),
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default=config.get('optimizer', 'optimizer'),
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=config.getfloat('optimizer', 'lr'), type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=config.getfloat('optimizer', 'momentum'), type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=config.getfloat('optimizer', 'weight_decay'), type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=config.getboolean('optimizer', 'nesterov'), type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default=config.get('scheduler', 'scheduler'),
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=config.getfloat('scheduler', 'min_lr'), type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=config.getfloat('scheduler', 'factor'), type=float)
    parser.add_argument('--patience', default=config.getint('scheduler', 'patience'), type=int)
    parser.add_argument('--milestones', default=config.get('scheduler', 'milestones'), type=str)
    parser.add_argument('--gamma', default=config.getfloat('scheduler', 'gamma'), type=float)
    parser.add_argument('--early_stopping', default=config.getint('early_stopping', 'early_stopping'), type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=config.getint('other', 'num_workers'), type=int)

    config = parser.parse_args()

    return config

"""
这个函数接受以下参数：
- `model`: 要训练的模型。
- `train_loader`: 训练数据集的数据加载器。
- `criterion`: 损失函数。
- `optimizer`: 优化器。
- `device`: 设备，可以是 `'cuda'` 或 `'cpu'`。
- `config`: 包含训练配置的字典，包括是否使用深度监督、学习率等超参数。

函数的主要步骤：
1. 将模型切换到训练模式：`model.train()`
2. 创建进度条对象，用于显示训练进度：`pbar = tqdm(total=len(train_loader))`
3. 遍历训练数据加载器中的每个样本：
   - 将输入数据和目标数据移动到设备上：`input = input.to(device)` 和 `target = target.to(device)`
   - 如果使用深度监督，通过模型进行前向传播，获取多个输出结果（列表），并计算损失和性能指标。
     - 遍历每个输出结果并计算损失：`for output in outputs: loss += criterion(output, target)`
     - 计算平均损失：`loss /= len(outputs)`
     - 计算最后一个输出结果和目标数据之间的 IoU 指标：`iou = iou_score(outputs[-1], target)`
   - 如果不使用深度监督，通过模型进行前向传播，获取单个输出结果，并计算损失：`output = model(input)` 和 `loss = criterion(output, target)`
   - 清零优化器的梯度缓冲区：`optimizer.zero_grad()`
   - 执行反向传播计算梯度：`loss.backward()`
   - 根据梯度更新模型的参数：`optimizer.step()`
   - 使用 `AverageMeter` 对象更新平均损失和平均 IoU：`avg_meters['loss'].update(loss.item(), input.size(0))` 和 `avg_meters['iou'].update(iou, input.size(0))`
   - 更新进度条的后缀信息：`postfix = OrderedDict([...])` 和 `pbar.set_postfix(postfix)`
   - 更新进度条的计数器：`pbar.update(1)`
4. 关闭进度条：`pbar.close()`
5. 返回包含平均损失和性能指标的有序字典：`return OrderedDict([...])`

这些步骤涵盖了在训练数据集上对模型进行一次迭代的训练过程。
"""
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


"""
这段代码定义了一个用于验证模型的函数 `validate()`。它接受以下参数：
- `config`：包含配置信息的字典。
- `val_loader`：验证数据集的数据加载器。
- `model`：要验证的模型。
- `criterion`：损失函数。

函数中的主要步骤如下：
1. 创建一个包含两个 `AverageMeter()` 对象的字典 `avg_meters`，用于计算并保存平均损失和平均 IoU（Intersection over Union）。
2. 将模型切换到评估模式，通过调用 `model.eval()` 来设置。
3. 使用 `torch.no_grad()` 上下文管理器，禁用梯度计算，以便在验证过程中不进行参数更新。
4. 使用 `tqdm` 进度条迭代 `val_loader` 中的每个验证样本。
5. 将输入数据和目标数据移动到设备（GPU 或 CPU）上。
6. 根据配置中的 `deep_supervision` 值，选择不同的计算输出和损失的方式：
   - 如果 `deep_supervision` 为真，则通过模型的多个输出计算损失，并取平均损失。
   - 如果 `deep_supervision` 为假，则使用模型的单个输出计算损失。
7. 使用输出和目标计算 IoU 值。
8. 更新平均损失和平均 IoU 的 `AverageMeter` 对象。
9. 设置进度条的后缀信息，包括平均损失和平均 IoU。
10. 更新并关闭进度条。
11. 返回一个有序字典，包含平均损失和平均 IoU 的键值对。

该函数的作用是计算模型在验证数据集上的损失和性能指标（IoU），并返回这些指标的平均值。
"""
def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def train_model():

    """
    这是一个将命令行参数解析为字典的操作，其中parse_args()是一个用于解析命令行参数的函数。
    vars()函数将解析出来的命令行参数对象转换为字典类型。
    这个操作的结果是将命令行参数以字典形式存储在config中。
    """
    config = vars(parse_args())

    """
    这段代码是为了创建用于保存模型的文件夹。首先判断配置中是否指定了模型名称，如果没有指定，则根据数据集和网络结构来生成默认的模型名称。
    如果配置中开启了深度监督（deep_supervision），则模型名称为"数据集_网络结构_wDS"；如果没有开启深度监督，则模型名称为"数据集_网络结构_woDS"。
    然后使用`os.makedirs`函数创建以模型名称和时间命名的文件夹，如果文件夹已存在则不会重复创建。
    """
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = f"models/{config['name']}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # 添加tensorBoard
    writer = SummaryWriter(f"{folder_name}/SummaryWriter")

    """
    这段代码会打印模型训练的配置信息。首先输出一条20个连字符"-"，作为分割线。
    然后遍历配置字典中的所有键值对，逐个输出键和对应的值，格式为"键: 值"。
    输出完后再次输出一条20个连字符"-"的分割线。
    这样可以方便用户在控制台上查看和核对模型训练的配置信息。
    """
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    """
    这段代码将模型的配置信息保存为YAML格式的文件。
    使用`open`函数打开一个文件，文件路径是根据模型名称组合而成的，路径为"models/模型名称/config.yml"。
    然后使用`yaml.dump`函数将配置信息写入文件中，以YAML格式保存。
    存入config.yml文件中
    这样可以方便后续查看和恢复模型训练的配置信息。
    """
    with open('%s/config.yml' % folder_name, 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    """
    这段代码根据配置中的损失函数类型来创建相应的损失函数实例。
        首先判断配置中的损失函数类型是否为'BCEWithLogitsLoss'，如果是，则使用`nn.BCEWithLogitsLoss()`来创建二分类交叉熵损失函数实例，并将其移动到指定的设备上。
        如果配置中的损失函数类型不是'BCEWithLogitsLoss'，则通过`losses.__dict__[config['loss']]()`来动态获取指定名称的损失函数类，并使用该类创建损失函数实例。然后将损失函数实例移动到指定的设备上。
    这样可以根据配置中的损失函数类型灵活地选择和使用不同的损失函数。
    """
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        # `losses.__dict__` 是一个字典，包含了当前上下文中所有可用的损失函数类。
        # `config['loss']` 是配置中指定的损失函数类型的字符串。
        # 通过 `losses.__dict__[config['loss']]` 可以获取到对应的损失函数类。
        # 然后使用 `()` 运算符创建该损失函数类的实例，并将其移动到指定的设备上（`to(device)`）。
        criterion = losses.__dict__[config['loss']]().to(device)

    """
    这行代码是用于设置cuDNN的benchmark模式为True。
    cuDNN（CUDA Deep Neural Network library）是一个针对深度神经网络计算的GPU加速库。
    cuDNN的benchmark模式可以自动寻找最适合当前硬件配置的卷积实现算法，并在训练过程中进行优化，提高运行效率。
    通过将`cudnn.benchmark`设置为True，可以启用cuDNN的benchmark模式，从而使得每次运行时都会重新评估和选择最佳的卷积实现算法，进而提高训练速度。
    需要注意的是，在某些情况下，benchmark模式可能会导致不确定性，因为每次运行时都会选择不同的实现算法。
        因此，如果要求结果的一致性比速度更重要，则可以将benchmark模式设置为False。
    而且：
        如果没有使用GPU加速，那么设置`cudnn.benchmark = True`不会产生任何效果，因为cuDNN库只能在GPU上使用。
        该代码的作用是启用cuDNN的benchmark模式来优化卷积计算，在使用GPU进行深度学习训练时可以提高训练速度。
        如果没有使用GPU，则无法享受到cuDNN带来的加速效果，即使设置了`cudnn.benchmark = True`也不会有任何效果。
    """
    if torch.cuda.is_available():
        cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # 通过 archs.__dict__[config['arch']] 可以获取到指定名称的模型结构类。
    # 然后调用该模型结构类的构造函数，并传入相应的参数：
    #   config['num_classes'] 表示分类任务的类别数，
    #   config['input_channels'] 表示输入数据的通道数，
    #   config['deep_supervision'] 表示是否使用深度监督（即是否在网络中添加多个分支用于不同层次的特征提取），
    # 并将创建好的模型实例赋值给变量 model。
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.to(device)

    # 这段代码是根据配置文件中指定的优化器和学习率调度器类型，创建对应的优化器和学习率调度器。
    params = filter(lambda p: p.requires_grad, model.parameters())# 对模型的参数进行过滤，只选择需要梯度更新的参数，并将其赋值给变量 params。
    """
    根据配置文件中的 config['optimizer'] 来选择使用哪种优化器。
        如果是 "Adam"，则创建一个 Adam 优化器，使用 optim.Adam() 函数，
            并传入相应的参数：params、lr=config['lr']（学习率）和 weight_decay=config['weight_decay']（权重衰减）。
        如果是 "SGD"，则创建一个 SGD 优化器，使用 optim.SGD() 函数，
            并传入相应的参数：params、lr=config['lr']、momentum=config['momentum']（动量）和 weight_decay=config['weight_decay']。
        如果既不是 "Adam" 也不是 "SGD"，则抛出一个 NotImplementedError 异常。
    """
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    """
    根据配置文件中的 config['scheduler'] 来选择使用哪种学习率调度器。
        如果是 "CosineAnnealingLR"，则创建一个 CosineAnnealingLR 调度器，使用 lr_scheduler.CosineAnnealingLR() 函数，
            并传入相应的参数：optimizer、T_max=config['epochs']（总的训练轮数）和 eta_min=config['min_lr']（学习率的最小值）。
        如果是 "ReduceLROnPlateau"，则创建一个 ReduceLROnPlateau 调度器，使用 lr_scheduler.ReduceLROnPlateau() 函数，
            并传入相应的参数：optimizer、factor=config['factor']（学习率缩放因子）、
                patience=config['patience']（在验证集上等待多少个epoch之后，学习率开始下降）、
                verbose=1（是否打印日志信息）和 min_lr=config['min_lr']（学习率的最小值）。
        如果是 "MultiStepLR"，则创建一个 MultiStepLR 调度器，使用 lr_scheduler.MultiStepLR() 函数，
            并传入相应的参数：optimizer、milestones=[int(e) for e in config['milestones'].split(',')]（在哪些epoch时学习率进行调整）
                和 gamma=config['gamma']（学习率缩放因子）。如果是 "ConstantLR"，则不使用学习率调度器，将 scheduler 设置为 None。
        如果既不是 "CosineAnnealingLR" 也不是 "ReduceLROnPlateau" 也不是 "MultiStepLR" 也不是 "ConstantLR"，则抛出一个 NotImplementedError 异常。
    """
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    """
    这行代码使用glob函数获取图像文件的路径。
        os.path.join('inputs', config['dataset'], 'images')用于生成图像文件的目录路径，
            其中'inputs'是根目录，config['dataset']是数据集名称，'images'是图像文件所在的子目录。
        '*' + config['img_ext']表示以任意字符开头，后面跟着图像文件扩展名。config['img_ext']是配置参数中指定的图像文件扩展名。
    最终，glob函数返回匹配指定模式的所有文件路径，并保存在img_ids列表中。每个文件的路径是相对于当前工作目录的路径。
    """
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    """
    这行代码对图像文件的路径进行处理，提取出图像文件的名称（去掉扩展名），并将名称保存在img_ids列表中。
    具体来说，代码通过遍历img_ids列表中的每个图像文件路径，
        使用os.path.basename(p)函数获取文件的基本名称（包括扩展名），
        然后使用os.path.splitext()函数将基本名称分割成文件名和扩展名的元组。
        由于我们只需要文件名部分，因此使用[0]索引获取文件名，并将文件名添加到img_ids列表中。
    最终，img_ids列表中保存的是图像文件的名称（不包含扩展名），用于后续的数据集划分或其他操作。
    """
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 将img_ids列表按照指定的比例划分为训练集和验证集，并将结果保存在train_img_ids和val_img_ids列表中。
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    # 数据增强：需要安装albumentations包
    train_transform = Compose([
        # 角度旋转
        albu.RandomRotate90(),
        # 图像翻转
        albu.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),  # 按照归一化的概率选择执行哪一个
        # 将图像大小调整为模型可接受的输入尺寸。
        albu.Resize(config['input_h'], config['input_w']),
        # 对图像进行归一化处理，将图像像素值缩放到0到1之间。
        albu.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    """
    通过Dataset类创建了训练集对象train_dataset。
        其中，img_ids参数是训练图像的名称列表（不包含扩展名），
        img_dir参数是训练图像文件存储的目录路径，
        mask_dir参数是训练标签（掩膜）文件存储的目录路径，
        img_ext参数是图像文件的扩展名，
        mask_ext参数是标签文件的扩展名，
        num_classes参数是数据集的类别数量，这里的分类数是不包括背景在内的。
        transform参数是对训练图像进行的数据增强变换操作流水线。
    """
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,# 表示在每个 epoch 开始时是否对数据进行洗牌，即打乱顺序。
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    """
    这段代码定义了一个有序字典 log，用于保存训练和验证过程中的一些指标值。
    log 字典中包含了以下键值对：
        'epoch'：用于保存每个 epoch 的编号。
        'lr'：用于保存每个 epoch 的学习率。
        'loss'：用于保存每个 epoch 的训练集损失。
        'iou'：用于保存每个 epoch 的训练集 Intersection over Union (IoU) 指标值。
        'val_loss'：用于保存每个 epoch 的验证集损失。
        'val_iou'：用于保存每个 epoch 的验证集 IoU 指标值。
    最后存入log.csv文件中
    """
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    start_time = time.time()
    best_iou = 0
    """
    trigger 的作用是用来判断是否触发早期停止（early stopping）。
        早期停止是一种常用的训练策略，它可以在验证集的性能不再提升时停止训练，以防止过拟合并节省计算资源。
    在这段代码中，
        trigger 的初始值为0。
        每当验证集的 IoU 指标不再提升时，trigger 的值就会增加1。
        如果 trigger 的值大于等于设定的早期停止阈值 config['early_stopping']，则会触发早期停止，训练循环会被中断，停止训练过程。
    通过使用 trigger 变量，可以在验证集性能不再提升时自动结束训练，避免过拟合并提高训练效率。
    """
    trigger = 0
    for epoch in range(config['epochs']):
        epoch = epoch + 1
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        writer.add_scalars("loss",
                           {"train_loss": train_log['loss'],
                            "test_loss": val_log['loss']},
                           epoch)
        writer.add_scalars("iou",
                           {"train_iou": train_log['iou'],
                            "test_iou": val_log['iou']},
                           epoch)

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        """
        这行代码将一个名为 `log` 的字典转换为 DataFrame，并将其保存为 CSV 文件。具体来说，它执行以下操作：
            1. 将 `log` 字典传递给 `pd.DataFrame()` 函数，将其转换为一个 DataFrame 对象。
            2. 使用 `to_csv()` 方法将 DataFrame 对象保存为 CSV 文件。
                - 参数 `'models/%s/log.csv' % config['name']` 是文件路径和名称的格式字符串，
                    其中 `%s` 会被 `config['name']` 的值替代。
                    例如，如果 `config['name']` 的值是 `"model1"`，那么保存的文件路径就是 `'models/model1/log.csv'`。
                - 参数 `index=False` 表示不将 DataFrame 的索引写入 CSV 文件。
        这段代码的目的是将训练过程中的日志信息保存为 CSV 文件，以便后续分析和可视化。
        """
        pd.DataFrame(log).to_csv('%s/log.csv' %
                                 folder_name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), '%s/model.pth' %
                       folder_name)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        """
        config['early_stopping']的默认值是-1，不满足第一个if条件，不触发早停策略。
        通常情况下，早期停止的阈值应该是一个非负整数，表示在多少个连续的验证集性能不再提升时触发停止训练。
            当将早停阈值设为-1时，意味着不使用早期停止策略，训练过程将一直进行下去，直到达到指定的训练轮数（config['epochs']）为止，或者手动停止训练。
        禁用早期停止可能会导致模型在训练过程中过拟合，因为它没有根据验证集的性能动态调整训练的停止时机。
        因此，建议在实际训练中根据需要设置合适的早停阈值，以避免过拟合和节省计算资源。
        """
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        end_time = time.time()
        print("本轮截至运行时间：",end_time - start_time)

        # 清显存，保证运行速率
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

train_model()





