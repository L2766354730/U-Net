import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

"""
这是一个 PyTorch 实现的损失函数模块，包括 `BCEDiceLoss` 和 `LovaszHingeLoss` 两个损失函数。

`BCEDiceLoss` 是一种结合了二元交叉熵 (Binary Cross Entropy, BCE) 和 Dice Loss 的损失函数。
BCE 用于度量预测值与真实值之间的差异，而 Dice Loss 则用于衡量两个集合之间的相似度。
相比于单一的 BCE 或 Dice Loss，BCEDiceLoss 可以更好地平衡准确率和召回率之间的权衡，从而提高模型的性能。

`LovaszHingeLoss` 则是一种基于 Lovasz 损失函数的损失函数。
Lovasz 损失函数是一种非常适合处理不平衡数据的损失函数，它可以在训练中强制模型对较难的样本进行更多的关注，从而提高模型的泛化能力。
LovaszHingeLoss 在二分类问题中表现得很好，特别是在处理分割问题时，可以取得很好的效果。

需要注意的是，`LovaszHingeLoss` 中的 `lovasz_hinge` 函数是在外部定义的，可能需要单独导入才能使用。
"""
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss