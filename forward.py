import numpy as np
from activation import activation


# 前向传播
def forward(x, W, activation_input, activation_hidden, activation_output):
    """
    x:  输入层的输入,单个样本\n
    W:  权重矩阵\n
    activation_input:  输入层的激活函数\n
    activation_hidden:  隐藏层的激活函数\n
    activation_output:  输出层的激活函数\n
    neural_elements:  神经元的输出,包括输入层，隐藏层，输出层\n
    """
    layers = len(W) + 1  # 层数
    hidden_elements = [x]
    for i in range(layers - 1):
        if i == 0:
            types = activation_input
        elif i == layers - 2:
            types = activation_output
        else:
            types = activation_hidden
        hidden_elements[i] = np.append(hidden_elements[i], -1)  # 添加偏置项
        ele = np.dot(hidden_elements[i], W[i])
        ele_i = activation(ele, types)
        # 上面的@表示矩阵乘法，加速运算可以使用np.dot:ele_i=
        hidden_elements.append(ele_i)
    neural_elements = hidden_elements

    return neural_elements



