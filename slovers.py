import numpy as np
import copy
from activation import activation_derivative
from Loss import loss, Dloss
from forward import forward


# 反向传播
def backward(
        X: np.ndarray,
        Y: np.ndarray,
        W: list,
        learning_rate: float,
        network_structure: list,
        activation_input: str,
        activation_hidden: str,
        activation_output: str,
):
    """
    x:输入层的输入（单个样本）\n
    y:输出层的输出（单个样本）\n
    """
    layers = len(W) + 1
    new_W = copy.deepcopy(W)
    # 计算LOSS
    LOSS = 0
    sample_num = X.shape[0]
    for sample in range(sample_num):
        x = X[sample, :]
        y = Y[sample, :]
        neural_elements = forward(x, W, activation_input, activation_hidden, activation_output)
        output = neural_elements[-1]
        LOSS = LOSS + loss(output, y)
    # 随机梯度下降
    # 随机在sample_num中选取十分之一的样本组成一个数组
    size = 0.2
    index = np.random.choice(sample_num,int(sample_num*size))
    for sample in index:
        x = X[sample, :]
        y = Y[sample, :]
        neural_elements = forward(x, W, activation_input, activation_hidden, activation_output)
        output = neural_elements[-1]
        # 反向传播
        # 1.求解各层的全微分矩阵
        Matrix = [Dloss(output, y)]

        for layer in range(layers-1, 0, -1):
            if layer == layers - 1:
                types = activation_output
            elif layer == 0:
                types = activation_input
            else:
                types = activation_hidden

            mat = np.ones((network_structure[layer], network_structure[layer-1]))
            for i in range(network_structure[layer]):
                for j in range(network_structure[layer-1]):

                    # 重点！！！！！
                    # 这里激活函数中的导数输入是求和
                    # b[l+1][i]=sum_{j}(W[l][j][i]*b[l][j])
                    b_l = 0
                    for t in range(network_structure[layer-1]+1):  # 包括偏置项
                        b_l = b_l + W[layer-1][t][i] * neural_elements[layer-1][t]
                    # j-->i
                    mat[i, j] = W[layer-1][j][i] * activation_derivative(b_l, types)

            Matrix.append(mat)
        # 此样本更新梯度
        grad_W = update_grad(W, neural_elements, Matrix, layers,
                             network_structure,
                             activation_input, activation_hidden, activation_output)

        # 更新权重
        for lens in range(layers - 1):
            '''
            if lens == 0:
                print("lens:\n",lens)
                print("W[lens]:\n",new_W[lens])
                print("learning_rate_init:\n",learning_rate_init)
                print("grad_W[lens]:\n",grad_W[lens])
            '''
            new_W[lens] = new_W[lens] - learning_rate * grad_W[lens]
            '''
            if lens == 0:
                print("new_W[lens]:\n",new_W[lens])
            '''
            if np.isnan(new_W[lens]).any():
                raise ValueError("权重矩阵中存在nan值，请检查学习率是否过大。")
    return LOSS, new_W
# 更新权重
def update_grad(
        W: list,
        neural_elements: list,
        Matrix: list,
        layers: int,
        network_structure: list,
        activation_input: str,
        activation_hidden: str,
        activation_output: str,
) -> list:
    # 计算梯度，grad_W[i]表示：第i层到第i+1层的权重矩阵的梯度
    grad_W = copy.deepcopy(W)
    for layer in range(layers - 1):
        if layer == layers - 2:
            types = activation_output
        elif layer == 0:
            types = activation_input
        else:
            types = activation_hidden
        for i in range(network_structure[layer + 1]):
            for j in range(network_structure[layer] + 1):
                num = layers - 1 - layer
                grad = 1
                for k in range(num):
                    grad = np.dot(grad, Matrix[k])
                grad = grad[i]
                # j-->i
                bl1 = 0
                for t in range(network_structure[layer] + 1):
                    bl1 = bl1 + W[layer][t][i] * neural_elements[layer][t]
                grad_W[layer][j][i] = grad * neural_elements[layer][j] * activation_derivative(bl1, types)

    return grad_W
