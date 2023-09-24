import numpy as np


def activation(x, activation_type):

    if activation_type == "sigmoid":
        # 范围：(0,1)
        return 1 / (1 + np.exp(-x))
    elif activation_type == "tanh":
        # 范围：(-1,1)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    elif activation_type == "relu":
        # 范围：(0,+∞)
        return np.maximum(0, x)

    elif activation_type == "leaky_relu":
        # 范围：(0,+∞)
        return np.maximum(0.01 * x, x)
    elif activation_type == "softmax":
        # 范围：(0,1)
        return np.exp(x) / np.sum(np.exp(x))
    elif activation_type == "equation":
        # 范围：(-∞,∞)
        return x
    elif activation_type == "linear":
        # 范围：(-∞,∞)
        return 0.1*x


# 激活函数的导数
def activation_derivative(x, activation_type):
    if activation_type == "sigmoid":
        return activation(x, activation_type="sigmoid") * (1 - activation(x, activation_type="sigmoid"))
    elif activation_type == "tanh":
        return 1 - activation(x, activation_type="tanh") ** 2
    elif activation_type == "relu":
        return np.where(x > 0, 1, 0)
    elif activation_type == "leaky_relu":
        return np.where(x > 0, 1, 0.01)
    elif activation_type == "softmax":
        return activation(x, activation_type="softmax") * (1 - activation(x, activation_type="softmax"))
    elif activation_type == "equation":
        return 1
    elif activation_type == "linear":
        return 0.1


if __name__ == "__main__":
    print("main")
