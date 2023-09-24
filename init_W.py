import numpy as np


def init_weight(X, Y, hidden_structure):
    if len(X.shape) == 1:
        # raise:抛出异常
        # 格式报错
        raise ValueError("输入错误：如果特征只有一个，请通过array.reshape(-1,1)将其转换为列向量。")
    if len(Y.shape) == 1:
        # 格式报错
        raise ValueError("输入错误：如果标签只有一个，请通过array.reshape(-1,1)将其转换为列向量。")
    if X.shape[0] != Y.shape[0]:
        # 格式报错
        raise ValueError("输入错误：特征和标签的行数不一致。")
    """
        初始化权重：
            W[k][i][j]:
                表示第k层到第k+1层的权重，其中i表示第k层的第i个节点，j表示第k+1层的第j个节点。
            k:0,1,2,...,layers-2
            i:0,1,2,...,nodes_{k}
            j:0,1,2,...,nodes_{k+1}
            接着，W中的每个元素也是
    """
    input_nodes = X.shape[1]
    output_nodes = Y.shape[1]

    network_structure = [input_nodes] + hidden_structure + [output_nodes]
    layers = len(network_structure)
    W = [
        np.random.randn(network_structure[i] + 1, network_structure[i + 1])
        for i in range(layers - 1)

    ]
    #预训练权重
    return W, network_structure
