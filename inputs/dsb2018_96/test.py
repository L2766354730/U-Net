import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import cv2

import archs


def maskShow():
    # 加载模型
    model = archs.NestedUNet(num_classes=1, input_channels=3, deep_supervision=False)
    model.load_state_dict(torch.load('../../models/dsb2018_96_NestedUNet_woDS_2024-01-15_22-50-34/model.pth'))
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载并预处理图像
    image_orgin = Image.open(
        "images/0e4c2e2780de7ec4312f0efcd86b07c3738d21df30bb4643659962b4da5505a3.png").convert('RGB')
    image = transform(image_orgin)
    image = image.unsqueeze(0)

    # 使用模型进行推理
    with torch.no_grad():
        output = model(image)

    # 将输出张量转换为图像
    output = output[0].squeeze().cpu().numpy()  # 将输出张量转换为 numpy 数组，并去掉 batch 和通道维度
    output = (output * 255).astype('uint8')  # 将数值范围从 [0, 1] 转换为 [0, 255] 并转换数据类型
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)  # 将三通道图像转换为单通道图像
    plt.imshow(output)
    plt.show()

    # 调整输出图像的大小与原始图像相同
    output = cv2.resize(output, (image_orgin.size[0], image_orgin.size[1]))


    # 处理输出结果
    image_orgin = np.array(image_orgin)
    img = cv2.bitwise_and(image_orgin, output)

    plt.title('Masks over image')
    plt.imshow(img)
    plt.show()

maskShow()