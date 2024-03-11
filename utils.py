import argparse


"""
这段代码包含了一些常用的辅助函数和类。

`str2bool`是一个用于解析命令行参数的函数，将字符串表示的布尔值转换为对应的Python布尔值。
    如果输入的字符串是"true"（不区分大小写）或1，则返回True；
    如果输入的字符串是"false"（不区分大小写）或0，则返回False；
    否则，抛出一个类型错误的异常。

`count_params`是一个用于计算模型参数数量的函数。
    它接受一个模型对象作为输入，并返回其中所有需要进行梯度计算的参数的总数量。
    通过遍历模型的parameters属性，并使用requires_grad属性来过滤出需要进行梯度计算的参数，
    然后使用numel方法获取每个参数的元素数量，并将其求和得到总数量。

`AverageMeter`是一个用于计算和存储平均值的类。
    它包含了val、avg、sum和count四个属性，分别表示当前值、平均值、值的累加和以及值的累加次数。
    reset方法用于重置所有属性的值为0，
    update方法用于更新当前值、累加和和累加次数，并计算新的平均值。

这些辅助函数和类可以在训练和评估深度学习模型时非常有用。
    例如，可以使用str2bool函数解析命令行参数中的布尔值选项，
    使用count_params函数统计模型的参数数量，
    使用AverageMeter类跟踪训练过程中的损失值或指标值的平均值。

"""
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count