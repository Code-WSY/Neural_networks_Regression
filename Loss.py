import numpy as np
# 损失函数：用来衡量预测值与真实值之间的差距
def loss(y_hat, y):
    return 0.5 * np.sum((y_hat - y) ** 2)
def Dloss(y_hat, y):
    diff = np.array(y_hat) - np.array(y)
    return diff
