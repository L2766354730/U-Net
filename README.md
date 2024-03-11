# 参考代码

[Python Unet ++ :医学图像分割，医学细胞分割，Unet医学图像处理，语义分割](https://blog.csdn.net/L_goodboy/article/details/130439416?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170453910116800184159706%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170453910116800184159706&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~sobaiduend~default-1-130439416-null-null.nonecase&utm_term=%E7%BB%86%E8%83%9E%E5%88%86%E5%89%B2pytorch&spm=1018.2226.3001.4450)

# 数据集
[100+医学影像数据集集锦](https://blog.csdn.net/qq_24662291/article/details/121183226)


# 改动
> 保证已有pytorch环境。

1. 上述链接代码导入pycharm

2. 安装  pip install -U albumentations  需要用命令，其他的就用pycharm自动安装即可

3. 将数据dsb-18放到inputs文件夹下，并修改preprocess_dsb2018.py中paths的路径
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c2b332703e63425c9d9d24d7e821a01f.png)

5. 运行preprocess_dsb2018.py文件，得到dsb2018_96文件夹，以及下面的已经处理好的数据

6. 将train.py文件中的cuda换成有cuda就cuda，没有就cpu的形式：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
```

5. 将train.py的main方法最后一行清空显存的cuda的语句（torch.cuda.empty_cache()）加上有cuda条件。
> cpu运行的话，如果换成以下cpu清内存，会发生错误就是内存中的被清空，反向传播计算的梯度无法保存，效果不提升。但是不清空速度会慢，但没有办法
```python
    # 删除所有全局变量
    for var in globals():
        del globals()[var]

    # 删除所有局部变量
    for var in locals():
        del locals()[var]

    # 触发垃圾回收器对不再使用的对象进行清理
    gc.collect()
```

6. 接着将unet++模型定义中的输出语句注释

7. 将epoch从100改为10，轮数太多运行太慢

8. 改显示的epoch
```python
   epoch = epoch + 1
   print('Epoch [%d/%d]' % (epoch, config['epochs']))
   ```

8. 将train.py的main方法中的cudnn.benchmark = True这句代码加上cuda可用条件


10. 将使用命令行参数的代码中的默认值使用ini配置文件中的值配置。这样可以使用ini配置文件改配置值，也可以使用命令行改配置。优先级是命令行大于ini。

11. 将保存的模型，在模型名称后加上时间，防止多次运行覆盖之前运行结果
   ```python
   import os
   import datetime
   
   timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
   folder_name = f"models/{config['name']}_{timestamp}"
   
   os.makedirs(folder_name, exist_ok=True)
   ```
11. 加入tensorboard：writer = SummaryWriter(f"{folder_name}/SummaryWriter")，可以可视化展示运行过程中iou、loss变化

 ```python
    writer.add_scalars("loss",
                       {"train_loss": train_log['loss'],
                        "test_loss": val_log['loss']},
                       epoch)
    writer.add_scalars("iou",
                       {"train_iou": train_log['iou'],
                        "test_iou": val_log['iou']},
                       epoch)
 ```
12. 加入时间控制：
    ```python
    start_time = time.time()
    end_time = time.time()
    print("本轮截至运行时间：",end_time - start_time)
    ```

12. 开始运行train.py

# 改动后代码及代码理解
1.  目录结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f6dc02920d924a5a87a59f19e7374395.png)
2. defaultValue.ini
3. archs.py
4. dataset.py
5. preprocess_dsb2018.py
6. losses.py
7. metrics.py
8. utils.py
9. train.py

# 接下来
- 对真正的测试集，也就是没有标签数据的测试集做预测，并生成掩膜，放到val.py
