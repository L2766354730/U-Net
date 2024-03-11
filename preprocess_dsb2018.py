import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

# 数据集预处理
def main():
    img_size = 96

    # glob这个函数是用来获取路径下所有文件的绝对路径，并组装成列表，以供后续使用
    paths = glob('inputs/dsb-18/stage1_train/*')

    """
        给这个文件夹生成目录树：
            tree:
                stage1_train
                ├── 0a7e06
                |    ├── images
                |       ├── 0a7e06.png
                |    └── masks
                |       ├── 0a7e06.png
                |       ├── 0aab0a.png
                |       ├── 0b1761.png
                ├── 0aab0a
                ├── 0b1761
                ├── ...
    """

    # 这里使用os库中的函数makedirs函数在文件夹不存在的情况下创建所给路径的文件夹
    os.makedirs('inputs/dsb2018_%d/images' % img_size, exist_ok=True)
    #  这里的 0 的意思是，本数据集只分一类，也就是只有背景和细胞两个分类，二分类。
    #  如果还有别的待分类的内容，就需要再建一个1的文件夹，这里面存的是二值图像关于这个类和背景的分割
    os.makedirs('inputs/dsb2018_%d/masks/0' % img_size, exist_ok=True)

    # 这里循环 每一个paths中的文件夹
    for i in tqdm(range(len(paths))):
        path = paths[i]
        # 取到路径中images文件夹中的图片
        # os.path.basename(path) ：返回一个文件路径的基名（即文件名）
        img = cv2.imread(os.path.join(path, 'images',
                                      os.path.basename(path) + '.png'))
        #  根据原始图像的形状创建全零图像，用于存储拼接后的masks图像
        mask = np.zeros((img.shape[0], img.shape[1]))
        for mask_path in glob(os.path.join(path, 'masks', '*')):
            # 将下属所有图像中值大于127，也就是二值图像中的255的白色部分，全部集合到mask图像中
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            mask[mask_] = 1

        # 这段代码的作用是处理图像的通道数：
        #       第一个条件len(img.shape) == 2判断图像是否是灰度图（通道数为1）。如果是灰度图，则使用np.tile()函数将图像在通道维度进行复制，使其变为RGB图像（通道数为3）。
        #       第二个条件img.shape[2] == 4判断图像是否具有4个通道（RGBA图像）。如果是RGBA图像，则取前3个通道，将其转换为RGB图像。
        # 这段代码的目的是确保图像的通道数为3，以便后续处理。如果图像是灰度图或RGBA图像，会根据需要进行相应的转换。
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]

        # 为方便后续训练，将图像与标签图像的大小resize为所要的大小。
        # 但是真实的UNet不是这样子的，真实的UNet用的是patch图像分块，并且还给图像进行了镜像padding
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        # 将处理好的图像存入指定文件夹中
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/images' % img_size,
                                 os.path.basename(path) + '.png'), img)
        # 这里要将mask*255，是因为mask中细胞位置是1，但是要显示为白色255，就需要都乘以255
        # 这种方式要记住
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/masks/0' % img_size,
                                 os.path.basename(path) + '.png'), (mask * 255).astype('uint8'))


if __name__ == '__main__':
    main()