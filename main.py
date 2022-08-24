import pickle

from dataset.mnist import load_mnist
import numpy as np
from PIL import Image


def img_show(img: np.ndarray):
    """
    用于展示训练的图像
    :param img:
    :return:
    """
    # 将Numpy数组的图像数组转换为PIL用的数据   因为读入图像时十一Numpy数组进行存储的
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


"""
load_mnist函数 以(训练图像,训练标签) (测试图像,测试标签)的形式读入MINIST数据
"""


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        # flatten参数: 设置是否展开输入图像
        # 设置为false时 则输入图像为1x28x28的三维数组
        # 设置为True则会将输入图像保存为784个元素构成的一维数组
        flatten=True,
        # normalize参数: 设置是否将输入图像正规化为0.0~1.0的值
        # 这种将数据限定在某个范围内的处理称为正规化处理   此外对神经网络的输入数据进行某种既定的转换的称为预处理
        # 设置为False 则图像会保持原来的0~255
        normalize=True,
        # one_hot_label参数:  设置是否将标签保存为one-hot
        # one_hot 表示是仅正确解标签为1 其余皆为0 的数组 如[0,0,0,1,0,1] 否则正确标签只是正常数值比如3 8等
        one_hot_label=False
    )
    return x_train, t_train, x_test, t_test


def init_network():
    """
    init_network() 会读取保存在pickle文件 sample_weight.pkl中的权重参数
    :return:
    """
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """
    将读入的数据进行处理 完成神经网络的推理处理
    :param network:
    :param x:
    :return:
    """

    def sigmoid(x:np.ndarray)->np.ndarray :
        return x / (1 + np.exp(-x))

    """
    使用softmax 作为输出函数 因为这是一个分类问题
    因为MINIST数据集是由0到9的数字图像构成的    而本次项目的就是识别手写的图像是0-9的数字类别
    """

    def softmax(a:np.ndarray)->np.ndarray :
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x_train, t_train, x, t = get_data()

# # 下面这段代码主要用于测试 数据是否正确读取 以及学习关于图像操作的一些知识
# img = x_train[0]
# label = t_train[0]
# print("label = {} img.shape = {}".format(label,img.shape))
# """
# 由于设置了 flatten=True 所以读入图像时是以一维NumPy数组的形式保存
# 因此显示图像时 需要把它变为原来的28x28
# """
# img = img.reshape(28,28)
#
# img_show(img)

network = init_network()
"""
对神经网络推理处理后 
需要评价它的识别精度 即能在多大程度上正确分类
"""
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    # 获取概率最高的元素的索引 即训练结果中识别正确最高的
    p = np.argmax(y)
    # 如果与测试集中的结果相同
    if p == t[i]:
        accuracy_cnt = accuracy_cnt + 1

print("Accuracy:{}".format(float(accuracy_cnt) / len(x)))
