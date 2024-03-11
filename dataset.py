import os

import cv2
import numpy as np
import torch
import torch.utils.data

# 数据集处理类
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids. 存着所有图像名称的列表
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension. image图像的类型
            mask_ext (str): Mask file extension. mask图像的类型
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # 读入image图像
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # 这里mask为列表，分几类，就有几-1个值.也就是每个类别对背景二分类。
        mask = []
        for i in range(self.num_classes):
            # 由于mask是灰度图，与使用灰度图的方式读入，但是使用[..., None]强制将灰度图像转换为具有额外维度的三维数组。
            # 这样做通常是为了与 RGB 图像具有相同的形状，以方便进行处理。
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])

        # np.dstack(mask)是将多个mask图像沿着深度维度进行堆叠，生成一个新的三维数组。
        # mask变量是一个二维列表，每个元素表示一类掩码图像，经过np.dstack()后，数组的深度维度就表示不同的掩码类别。
        # 也就是：转置卷积和1*1卷积综合作用后生成大小和原图一致，但是通道数为分类类别数。每个通道中每个像素的值是一个概率，代表这个像素属于该通道的概率。
        # 这里就是生成了类别数个通道，每个通道代表每个像素分到这个通道的概率，但是这里是255
        mask = np.dstack(mask)

        """
             这段代码是使用`albumentations`库对图像进行数据增强。
               `albumentations`库是一个开源的图像增强库，能够在训练深度学习模型时为图像增加随机变换和扰动，从而提高模型的泛化能力。
                在这段代码中，如果定义了`transform`参数（即数据增强的变换），则会使用该参数对图像和掩模进行增强。
            
             如果要使用数据增强，`transform`参数应该传入一个`albumentations`库中定义的变换函数或变换组合。
                `albumentations`库提供了多种各具特色的图像增强操作，例如随机裁剪、旋转、缩放、翻转、色彩调整等。
                 这些增强操作可以单独使用，也可以通过`Compose`函数组合在一起形成一个变换组合。
    
            下面是一个示例，展示如何使用`Compose`函数将多个增强操作组合在一起：
    
            ```python
            import albumentations as A
    
            transform = A.Compose([
                A.RandomCrop(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
            ```
    
            在上述示例中，`RandomCrop`表示随机裁剪，`HorizontalFlip`表示水平翻转，`ColorJitter`表示颜色调整。
            这些操作会按照指定的概率（例如`p=0.5`）对输入图像进行随机变换。
            然后，可以将`transform`作为参数传递给图像增强代码中的`self.transform`，以应用相应的数据增强操作。
            
            具体地，`self.transform(image=img, mask=mask)`会将输入的原始图像和掩模传递给变换函数进行增强，返回增强后的图像和掩模。
            `augmented`是一个字典，包含了增强后的图像和掩模。
            `img = augmented['image']`和`mask = augmented['mask']`用于分别获取增强后的图像和掩模，并将其赋值回原来的变量。
    
            需要注意的是，`albumentations`库处理的图像和掩模都是`numpy`数组格式，且通道顺序为RGB。PyTorch以及OpenCV中通道都是RGB
        """
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)  # 这个包比较方便，能把mask也一并做掉
            img = augmented['image']  # 参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']

        # 这段代码对图像和掩模进行了一些预处理操作。
        # 将图像的数据类型转换为float32，并将像素值归一化到[0, 1]的范围。
        # 改变图像的维度顺序，将通道维度放在最前面，以适应深度学习框架的输入要求。
        # 同样地，对掩模也进行了类似的处理，和神经网络预测的结果一样，要转换为0-1的范围内，也就是概率值，也就是最重要预测的值。
        # 对图像归一化的作用：
        # 1. 对图像进行归一化可以使得图像的像素值在[0,1]范围内，这有助于优化模型的训练和收敛速度。
        #     具体来说，归一化可以使得数据的均值接近于0，方差接近于1，这对于许多基于梯度的优化算法（如随机梯度下降）是非常有利的。
        # 2. 此外，由于神经网络的权重通常会被初始化为较小的随机值，因此在训练过程中，输入的像素值可能会因为网络权重较小而产生较小的响应，
        #     这可能导致网络学习缓慢或停滞不前。通过对输入图像进行归一化，可以缓解这个问题，使网络更容易学习到有用的特征。
        # 3. 另外，还可以防止图像数据因为数值范围过大而在计算中出现溢出或浮点数精度丢失等问题。
        img = img.astype('float32') / 255

        # 在深度学习中，通常要求输入的图像或掩模的维度顺序为通道维度在最前面，即`(channels, height, width)`。
        # 这种维度顺序可以方便地与卷积操作等深度学习操作相匹配。
        # 在原始的`mask`中，假设维度顺序为`(height, width, channels)`，也确实是这样。通过进行维度变换 `mask = mask.transpose(2, 0, 1)`，
        # 就将通道维度放在了最前面，符合深度学习框架对输入数据的要求。
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        # 返回三个数据：图像、掩膜、图像名称。最后调用该类会将所有的组合成一个列表返回。
        return img, mask, {'img_id': img_id}


    """
    fcn的用的是argmax，将所有通道的这个像素值中最大的索引取出来，代表值，于是fcn就用的标签数组。
    而这里的mask就是所有的通道，于是就不能用argmax了。
    """

    """
    # 展示多张图片
    def show_images(imgs, num_rows, num_cols, scale=2):
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i in range(num_rows):
            for j in range(num_cols):
                axes[i][j].imshow(imgs[i * num_cols + j])
                axes[i][j].axes.get_xaxis().set_visible(False)
                axes[i][j].axes.get_yaxis().set_visible(False)
        plt.show()
        return axes
    """

    """
    掩膜图像展示法
    img = cv2.bitwise_and(train[0][2],cv2.cvtColor(train[1][2], cv2.COLOR_GRAY2BGR))
    plt.title('Masks over image')
    plt.imshow(img)
    plt.show()
    """