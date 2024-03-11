import torch
from torch import nn

__all__ = ['UNet', 'NestedUNet']

"""
- 这是一个 VGG 网络中常用的基本块类 VGGBlock 的定义，它包含两个卷积层和两个批归一化层，其中 ReLU 激活函数被应用在每个卷积层之后。
- 这个类的构造函数 __init__ 接受三个参数：输入通道数 in_channels，中间通道数 middle_channels 和输出通道数 out_channels。在构造函数中，首先调用父类的构造函数 super().__init__() 进行初始化。然后，定义了卷积层 self.conv1 和 self.conv2，分别将输入通道数变换为中间通道数和输出通道数。同时，两个卷积层的卷积核大小都是 3x3，**填充大小为 1，以保持特征图的尺寸不变**。接着，定义了批归一化层 self.bn1 和 self.bn2，用于规范化卷积层的输出。最后，定义了 ReLU 激活函数 self.relu。
- forward 方法是前向传播函数，在该方法中实现了网络层的前向计算逻辑。首先通过第一个卷积层和批归一化层处理输入 x，然后应用 ReLU 激活函数。接着，将结果传递给第二个卷积层和批归一化层，再次应用 ReLU 激活函数。最后，将输出结果返回。
- 这个类定义了一个 VGG 网络中常用的基本块，可以在 UNet 网络的不同层中多次调用这个块来构建整个网络。
"""
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

"""
- 这段代码定义了一个名为 `UNet` 的类，继承自 `nn.Module` 类。该类是一个 U-Net 网络，用于图像分割任务。它的输入是一个大小为 `input_channels` 的图像，输出是一个大小为 `num_classes` 的分割图像。
- 在类的初始化函数中，定义了一个包含 5 个元素的列表 `nb_filter`，表示每个卷积层的输出通道数。然后创建了 `MaxPool2d` 和 `Upsample` 对象，用于网络中的最大池化和上采样操作。接下来创建了 5 个卷积块对象 `conv0_0` 到 `conv4_0`，每个卷积块包含两个卷积层和一个 ReLU 层。其中，第一个卷积层的输入通道数为 `input_channels` 或前一层卷积层的输出通道数，输出通道数为 `nb_filter[0]` 到 `nb_filter[4]`，依次递增。第二个卷积层的输入通道数等于第一个卷积层的输出通道数，输出通道数仍然为 `nb_filter[0]` 到 `nb_filter[4]`。这些卷积块的作用是提取不同尺度的特征信息。
- 接下来创建了 4 个卷积块对象 `conv3_1` 到 `conv0_4`。这些卷积块的输入通道数分别为 `nb_filter[3]+nb_filter[4]`、`nb_filter[2]+nb_filter[3]`、`nb_filter[1]+nb_filter[2]` 和 `nb_filter[0]+nb_filter[1]`，输出通道数仍然为 `nb_filter[0]`。这些卷积块的作用是将不同尺度的特征信息进行融合，以便更好地进行分割。
- 最后定义了一个卷积层 `final`，用于生成分割结果。它的输入通道数等于最后一个卷积块的输出通道数，输出通道数为 `num_classes`。在 `forward()` 函数中，先通过各个卷积块提取特征信息，然后将不同尺度的特征信息进行融合，最终通过卷积层 `final` 生成分割结果。
- 在这里，`1` 是 `torch.cat()` 函数的第二个参数 `dim` 的值。`torch.cat()` 函数用于沿着指定的维度将张量进行拼接。
- 在这个特定的语句中，`torch.cat([x3_0, self.up(x4_0)], 1)` 表示将 `x3_0` 和 `self.up(x4_0)` 沿着第一个维度（维度索引从0开始）进行拼接。通过使用 `1` 作为 `dim` 参数的值，表示沿着第一个维度进行拼接。
- 这里的目的是将 `x3_0` 和上采样后的 `x4_0` 进行拼接，以便在 U-Net 网络中进行特征融合操作。拼接后的结果将作为输入传递给 `self.conv3_1` 卷积块进行处理。
"""
"""
测试网络正确性代码
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 测试网络的正确性
input = torch.ones((2,3,400,400))
unet =UNet(1).to(device)  
output = unet(input)
print(output.shape)

- 这里出现了问题：
    - 当测试图片大小为572，572时：
        1. torch.Size([2, 32, 572, 572])
        2. torch.Size([2, 64, 286, 286])
        3. torch.Size([2, 128, 143, 143])
        4. torch.Size([2, 256, 71, 71])
        5. torch.Size([2, 512, 35, 35])
        6. torch.Size([2, 512, 70, 70])
- 这是因为：
    - 下采样71是单数，变到了35，而上采样35乘2是70
    - x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  这个加法没办法加
"""
"""
torch.Size([2, 32, 400, 400])
torch.Size([2, 64, 200, 200])
torch.Size([2, 128, 100, 100])
torch.Size([2, 256, 50, 50])
torch.Size([2, 512, 25, 25])
torch.Size([2, 256, 50, 50])
torch.Size([2, 128, 100, 100])
torch.Size([2, 64, 200, 200])
torch.Size([2, 32, 400, 400])
torch.Size([2, 1, 400, 400])
"""
class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

"""
- conv0_0: 前面的0代表第0层。后面的0代表第0列，斜着的列
- unet++中的通道数: 每一层的输出通道数都一致，输入通道数为前面密集连接拼起来的和
"""
"""
# 测试网络的正确性
input = torch.ones((2,3,400,400))
nest_unet =NestedUNet(1, deep_supervision=True).to(device)  
output = nest_unet(input)
for result in output:
    print("out: ",result.shape)
"""
"""
input: torch.Size([2, 3, 400, 400])
x0_0: torch.Size([2, 32, 400, 400])
x1_0: torch.Size([2, 64, 200, 200])
x0_1: torch.Size([2, 32, 400, 400])
x2_0: torch.Size([2, 128, 100, 100])
x1_1: torch.Size([2, 64, 200, 200])
x0_2: torch.Size([2, 32, 400, 400])
x3_0: torch.Size([2, 256, 50, 50])
x2_1: torch.Size([2, 128, 100, 100])
x1_2: torch.Size([2, 64, 200, 200])
x0_3: torch.Size([2, 32, 400, 400])
x4_0: torch.Size([2, 512, 25, 25])
x3_1: torch.Size([2, 256, 50, 50])
x2_2: torch.Size([2, 128, 100, 100])
x1_3: torch.Size([2, 64, 200, 200])
x0_4: torch.Size([2, 32, 400, 400])
out:  torch.Size([2, 1, 400, 400])
out:  torch.Size([2, 1, 400, 400])
out:  torch.Size([2, 1, 400, 400])
out:  torch.Size([2, 1, 400, 400])
"""
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # print('input:', input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:', x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:', x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:', x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:', x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:', x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:', x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:', x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:', x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:', x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:', x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:', x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:', x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:', x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:', x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:', x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output