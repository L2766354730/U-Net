import numpy as np
import torch
import torch.nn.functional as F

"""
这段代码定义了两个评估指标函数：`iou_score`和`dice_coef`。这些指标通常用于评估图像分割模型的性能。

`iou_score`计算了预测输出（output）和目标标签（target）之间的交并比（Intersection over Union，IoU）。
    首先，通过对预测输出应用sigmoid函数将其转换为概率值，并使用`torch.sigmoid()`函数将其转换为[0,1]范围内的值。
    然后，将预测输出和目标标签转换为numpy数组。
    接下来，根据阈值0.5将预测输出和目标标签二值化为True/False表示。
    然后，计算交集和并集的元素数量，并将其相加得到交并比。
    为了避免除以0的情况，添加了一个平滑项smooth。

`dice_coef`计算了预测输出和目标标签之间的Dice系数。
    首先，对预测输出应用sigmoid函数并将其视为一维数组。
    然后，将预测输出和目标标签转换为numpy数组。
    接下来，计算预测输出和目标标签之间的交集元素的和，并将其乘以2。
    然后，计算预测输出和目标标签中各自元素的和，并将其相加。
    最后，将交集和平滑项添加到分子中，将预测输出和目标标签的和以及平滑项添加到分母中，得到Dice系数。

这段代码使用了PyTorch库的一些函数，如`torch.sigmoid`、`torch.is_tensor`和`torch.Tensor.view`，
并使用了NumPy库的函数，如`numpy.sum`和`numpy.ndarray`。
如果要完整运行该段代码，需要确保已正确导入相关库，并提供合适的输出和目标标签张量作为输入。
"""
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)