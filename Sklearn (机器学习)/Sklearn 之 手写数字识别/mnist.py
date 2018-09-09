import os
import numpy as np

from sklearn.neural_network import MLPClassifier

'''
def load_data(image, label):
    """
    加载数据
    """
    with gzip.open(os.path.join('./data', label)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(os.path.join('./data', image), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image, label
train_dataSet, train_hwLabels = load_data('train-images-idx3-ubyte.gz',
                                          'train-labels-idx1-ubyte.gz')

test_dataSet, test_hwLabels = load_data('t10k-images-idx3-ubyte.gz',
                                        't10k-labels-idx1-ubyte.gz')
'''

def img2vector(filename):
    """
    定义img2vector函数，将加载的32*32 的图片矩阵展开成一列向量
    """
    retMat = np.zeros([1024], int)
    with open(filename) as fr:
        lines = fr.readlines()
    for i in range(32):
        for j in range(32):     # 将01数字存放在retMat
            retMat[i*32+j] = lines[i][j]
    return retMat


def readDataSet(path):
    fileList = os.listdir(path)         # 获取文件夹下所有文件
    numFiles = len(fileList)            # 统计需要读取的文件的数目
    dataSet = np.zeros([numFiles, 1024], int)   # 用于存放所有的数字文件
    hwLabels = np.zeros([numFiles, 10])         # 用于存放对应的标签one-hot
    for i in range(numFiles):
        filePath = fileList[i]          # 获取文件名称/路径
        digit = int(filePath.split('_')[0])
        hwLabels[i][digit] = 1.0
        dataSet[i] = img2vector(os.path.join(path, filePath))   # 获取文件内容
    return dataSet, hwLabels

train_dataSet, train_hwLabels = readDataSet('./data/dbrhd/pendigits-orig.tes.Z')


# 构建神经网络：设置网络的隐藏层数、各隐藏层神经元个数、
# 激活函数、学习率、优化方法、最大迭代次数
# hidden_layer_sizes 存放的是一个元组，表示第i层隐藏层里神经元的个数
# 使用logistic激活函数和adam优化方法，并令初始学习率为0.0001
clf = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic',
                    solver='adam', learning_rate_init=0.0001, max_iter=2000)

# fit函数能根据训练集及对应标签集自动设置感层机的输入与输出层的神经元个数
# 例如train_dataSet为n*1024的矩阵，train_hwLabels为n*10的矩阵
# 则fit函数将MLP的输入层神经元个数设为1024，输出层神经元个数为10
clf.fit(train_dataSet, train_hwLabels)

# 测试集评价
test_dataSet, test_hwLabels = readDataSet('./data/dbrhd/pendigits-orig.tra.Z')
res = clf.predict(test_dataSet)  # 对测试集进行预测
error_num = 0   # 统计预测错误的数目
num= len(test_dataSet)   # 测试集的数目
for i in range(num):
    # 比较长度为10的数组，返回包含01的数组，0为不同，1为相同
    if np.sum(res[i] == test_hwLabels[i]) < 10:
        error_num += 1
print("Total num: ", num,
      "Wrong num: ", error_num,
      "Wrong Rate: ", error_num/float(num))