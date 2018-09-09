# Sklearn 之手写数字识别
*** 使用sklearn库里的分类模型来对手写数字（MNIST）做分类实践 ***

## 数据源

MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST). 训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据.
MNIST 数据集可在 http://yann.lecun.com/exdb/mnist/ 获取, 它包含了四个部分:

- Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
- Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
- Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
- Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)

[DBRHD（Pen-Based Recognition of Handwritten Digits Data Set）](http://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/)
是UCI的机器学习中心提供的数字手写体数据库:
DBRHD数据集包含大量的数字0~9的手写体图片，这些图片来源于44位不同的人的手写数字，图片已归一化为以手写数字为中心的32*32规格的图片。

DBRHD的训练集与测试 集组成如下：

- 训练集：7,494个手写体图片及对应标签，来源于40位手写者 

- 测试集：3,498个手写体图片及对应标签，来源于14位手写者