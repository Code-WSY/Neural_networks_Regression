import matplotlib.pyplot as plt
import time
import numpy as np


class Neural_networks:
    def __init__(self):
        '''
                   初始化参数：
                   1.自定义隐藏层结构：hidden_structure
                   2.总层数（包括输入和输出层）：layers
                   3.初始学习率：learning_rate
                   4.学习率衰减指数：power_t
                   5.学习率增加指数:increase_t
                   6.grad_explosion_t: 梯度爆炸时，学习率的衰减指数
                   7.损失函数：Loss
                   8.最大迭代次数：max_epochs
                   9.随机梯度法的采样比例：mini_batch_size
                   10.输入层到隐藏层的激活函数：activation_input
                   11.隐藏层之间的激活函数：activation_hidden
                   12.隐藏层到输出层的激活函数：activation_output
                   13.连续n_iter_nochange次损失函数增加，停止迭代
                   14.min_loss:当损失函数的值小于min_loss时，就停止迭代
                   15.tol: 损失函数的值变化阈值tol
                   16.scf：当损失函数的值变化连续scf次小于tol这个值时，就停止迭代
                   17.n_iter：迭代次数
               '''
        self.hidden_structure = [20,20,20]
        self.layers = len(self.hidden_structure) + 2
        self.learning_rate_init = 0.5
        self.power_t = 0.5
        self.increase_t=1.05
        self.grad_explosion_t=0.5
        self.Loss = 0
        self.max_epochs = 200
        self.mini_batch_size = 0.1
        self.activation_input = "relu"
        self.activation_hidden = "tanh"
        self.activation_output = "equation"
        self.n_iter_nochange=10
        self.min_loss=1e-3 #
        self.tol=1e-5
        self.scf = 10
        self.n_iter=0
        #缩放系数
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ):
        from init_W import init_weight
        from backward import backward, update_grad
        hidden_structure=self.hidden_structure
        learning_rate= self.learning_rate_init / X.shape[0]
        epochs=self.max_epochs
        activation_input=self.activation_input
        activation_hidden=self.activation_hidden
        activation_output=self.activation_output
        """
        X:输入层的输入
        Y:输出层的输出
        """
        # 初始化权重
        W, network_structure = init_weight(X, Y, hidden_structure)
        Loss_epoch = []
        sample_num = X.shape[0]
        count_increase = 0 #记录连续增加的次数
        count_scf=0 #记录自洽的次数
        for i in range(self.max_epochs):
            self.n_iter+=1
            Loss, W = backward(
                X,
                Y,
                W,
                learning_rate,
                self.mini_batch_size,
                network_structure,
                activation_input,
                activation_hidden,
                activation_output,
            )
            #记录连续增加的次数，调整学习率
            if Loss>self.Loss:
                count_increase+=1
                # 当连续增加的次数大于等于n_iter_nochange时，停止迭代
                if count_increase >= self.n_iter_nochange:
                    print("连续{}次损失函数增加，停止迭代。".format(self.n_iter_nochange))
                    break
                learning_rate=learning_rate*self.power_t
            else:
                count_increase=0
                learning_rate=learning_rate*self.increase_t

            # 当损失函数的值小于tol时，就不会继续迭代了
            if  Loss<self.min_loss:
                print('此次迭代损失函数的值为:{}，上次迭代损失函数的值为:{}'.format(Loss,self.Loss))
                print("损失函数误差小于{},停止迭代".format(self.min_loss))
                Loss_epoch.append(Loss)
                break

            if abs(self.Loss-Loss)<self.tol and Loss>self.min_loss:
                count_scf+=1
                Loss_epoch.append(Loss)
                if count_scf==self.scf:
                    print('损失函数的值变化小于{}已连续达{}次，停止迭代。'.format(self.tol,self.scf))
                    break
            else:
                count_scf=0
                Loss_epoch.append(Loss)

            # 当损失函数的值大于1e10时，说明梯度爆炸，需要调小学习率
            if abs(self.Loss-Loss) >1e10 and count_increase>=2:
                print("梯度爆炸，程序将自动调小学习率并重新初始化权重。")
                learning_rate=learning_rate*self.grad_explosion_t
                W, network_structure = init_weight(X, Y, hidden_structure)
                #重新初始化权重后，将连续增加的次数清零,将损失函数的值清零
                self.Loss=0
                count_increase=0
                count_scf = 0
                Loss_epoch=[]
            else:
                self.Loss = Loss
                Loss_epoch.append(Loss)


            if i % 10 == 0:
                print("epoch:", i, "Loss:", Loss)
        self.W = W
        # 找出Loss_epoch中第一个小于1的值，然后将其之前的值全部删除
        for i in range(len(Loss_epoch)):
            if Loss_epoch[i] < 1:
                Loss_epoch = Loss_epoch[i:]
                break
        return Loss_epoch
    # 预测
    def predict(self, X):
        from forward import forward
        y_hat = []
        #先将X压缩到-1,1之间
        #X=X/self.scaleX
        for sample in range(X.shape[0]):
            x = X[sample, :]
            neural_elements = forward(
                x,
                self.W,
                activation_input=self.activation_input,
                activation_hidden=self.activation_hidden,
                activation_output=self.activation_output,
            )

            #将Y值解压缩
            y_hat.append(neural_elements[-1])
        y_hat = np.array(y_hat)
        return y_hat

if __name__ == "__main__":
    nn = Neural_networks()
    np.random.seed(0)
    def f(X):
        return 3*np.sin(2*X)+2*np.cos(3*X)+X**2+X+1
    sample_num =500
    xlim = [-2, 2]
    X = np.linspace(xlim[0], xlim[1], sample_num).reshape(-1, 1)
    Y = f(X).reshape(-1, 1)+np.random.normal(0,0.1,sample_num).reshape(-1,1)
    #划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    Loss = nn.fit(X_train, Y_train)
    print("一共迭代了{}次".format(nn.n_iter))
    print("power_t:", nn.power_t)
    y_hat = nn.predict(X_test)
    #输出参数
    print("hidden_structure:", nn.hidden_structure)
    print("learning_rate_init:", nn.learning_rate_init)
    print("sample_num:", sample_num)
    # 画图
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    #TypeError: list indices must be integers or slices, not tuple
    plt.plot(X, Y, c="b", label="Real")
    plt.scatter(X_test, y_hat, c="r", s=10, alpha=0.5, label="Predict", marker="o")
    plt.legend()
    plt.title("Neural Networks Fit")
    plt.subplot(1, 2, 2)
    plt.plot(Loss, c="black", label="Loss", linewidth=2)
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("Neural Networks Fit.png")
    plt.show()