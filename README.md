# 图像分类
## 使用残差网络进行图像分类
## 原有代码仓库地址为[https://github.com/pytorch/examples/tree/master/imagenet]

## 图像分类基础流程
- 获取数据集（训练集、验证集、测试集），并将数据集中的每张图片赋予类别标签。如数据集中有5类图片，可用1~5进行类别标识
- 使用代码构建数据集加载器，带后续训练测试事进行数据的加载
- 使用数据增强器进行数据的增强，以提高模型的泛化能力
- 主干网络可以任意搭建，目标在于更好的提取图像的特征，以用于分类
- 最后一层输出层，一般来说使用全连接层来进行分类，即有多少类别，fc就输出多少维度，最后用max来提取张量中具有最大值的index，用该index来判断类别
