import matplotlib.pyplot as plt
import networkx as nx


class Graph_Neural_Networks:
    def __init__(self):
        # 输入节点数
        self.input_nodes = 3
        # 隐藏节点数
        self.hidden_nodes = 4
        # 隐藏层数
        self.num_layers = 2
        # 输出节点数
        self.output_nodes = 2

    def Graph_Neural_Networks(
        self, input_nodes=4, hidden_nodes=6, num_layers=4, output_nodes=3
    ):
        """
        Args:
            input_nodes: 输入节点数
            hidden_nodes: 每层的隐藏节点数
            num_layers:  隐藏层数
            output_nodes:  输出节点数
        Returns:
            输出一个有向图
        """

        # 创建一个有向图，大小
        G = nx.DiGraph()
        # 设置图片大小
        plt.figure(figsize=(15, 10))
        # 添加输入节点
        for i in range(input_nodes):
            G.add_node(f"Input{i + 1}")
        # 添加隐藏节点和边
        for layer in range(num_layers):
            for i in range(hidden_nodes):
                G.add_node(f"Hidden{layer + 1}_{i + 1}")
                if layer == 0:
                    for j in range(input_nodes):
                        G.add_edge(f"Input{j + 1}", f"Hidden{layer + 1}_{i + 1}")
                else:
                    for j in range(hidden_nodes):
                        G.add_edge(
                            f"Hidden{layer}_{j + 1}", f"Hidden{layer + 1}_{i + 1}"
                        )
        # 添加输出节点和边
        for i in range(output_nodes):
            G.add_node(f"Output{i + 1}")
            for j in range(hidden_nodes):
                G.add_edge(f"Hidden{num_layers}_{j + 1}", f"Output{i + 1}")
        # 设置节点位置
        pos = {}
        input_up = (hidden_nodes - input_nodes) / 2
        output_up = (hidden_nodes - output_nodes) / 2
        # 设置输入节点的位置
        for i in range(input_nodes):
            pos[f"Input{i + 1}"] = (0, i + input_up)  # 0表示第0层，i表示第i个节点
        # 设置隐藏节点的位置
        for layer in range(num_layers):
            for i in range(hidden_nodes):
                pos[f"Hidden{layer + 1}_{i + 1}"] = (layer + 1, i)
        # 设置输出节点的位置
        for i in range(output_nodes):
            pos[f"Output{i + 1}"] = (num_layers + 1, i + output_up)

        # ------------------------绘制输入层到隐藏层边的格式------------------------#
        font_size = 10
        node_size = 3000
        # 输入节点的颜色，大小，形状，透明度，边框颜色，边框宽度
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[f"Input{i + 1}" for i in range(input_nodes)],  # lat
            node_color="red",
            node_size=node_size,
            node_shape="o",
            alpha=0.3,
            edgecolors="black",
            linewidths=1,
        )
        # 设置输入标签的颜色，大小，字体,显眼的字体,latex字体:x_{1}用代码为：$x_{1}$,引号前面加r表示不转义，latex需要加r
        nx.draw_networkx_labels(
            G,
            pos,
            labels={f"Input{i + 1}": f"$X_{i + 1}$" for i in range(input_nodes)},
            font_color="black",
            font_size=font_size,
            font_family="Times New Roman",
        )
        # 绘制输入层到隐藏层边的颜色，宽度，样式
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[
                (f"Input{i + 1}", f"Hidden1_{j + 1}")
                for i in range(input_nodes)
                for j in range(hidden_nodes)
            ],
            edge_color="red",
            width=1,
            style="dashed",
        )
        # 在边上添加权重
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={
                (f"Input{i + 1}", f"Hidden1_{j + 1}"): f"$w_{{0{i + 1}1{j + 1}}}$"
                for i in range(input_nodes)
                for j in range(hidden_nodes)
            },
            font_size=font_size,
            font_family="Times New Roman",
        )
        # ------------------------------------------------------------------------#

        # --------------------------绘制隐藏层到隐藏层边的格式-------------------------#
        # 设置隐藏节点的格式
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[
                f"Hidden{layer + 1}_{i + 1}"
                for layer in range(num_layers)
                for i in range(hidden_nodes)
            ],
            node_color="blue",
            node_size=node_size,
            node_shape="o",
            alpha=0.4,
            edgecolors="black",
            linewidths=1,
        )
        # 设置隐藏标签的格式: x_{1,2}
        nx.draw_networkx_labels(
            G,
            pos,
            labels={
                f"Hidden{layer + 1}_{i + 1}": f"$H_{{{layer + 1}{i + 1}}}$"
                for layer in range(num_layers)
                for i in range(hidden_nodes)
            },
            font_color="black",
            font_size=font_size,
            font_family="Times New Roman",
        )
        # 绘制隐藏层到隐藏层边的格式
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[
                (f"Hidden{layer}_{j+1}", f"Hidden{layer + 1}_{i+1}")
                for layer in range(1, num_layers)
                for i in range(hidden_nodes)
                for j in range(hidden_nodes)
            ],
            edge_color="blue",
            width=1,
            style="dashed",
        )
        # ------------------------------------------------------------------------#

        # --------------------------绘制隐藏层到输出层边的格式-------------------------#
        # 设置输出节点的格式
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[f"Output{i + 1}" for i in range(output_nodes)],
            node_color="green",
            node_size=node_size,
            node_shape="o",
            alpha=0.4,
            edgecolors="black",
            linewidths=1,
        )
        # 设置输出标签的格式
        nx.draw_networkx_labels(
            G,
            pos,
            labels={f"Output{i + 1}": f"$y_{i + 1}$" for i in range(output_nodes)},
            font_color="black",
            font_size=font_size,
            font_family="Times New Roman",
        )
        # 绘制隐藏层到输出层 边 的格式
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[
                (f"Hidden{num_layers}_{j + 1}", f"Output{i + 1}")
                for i in range(output_nodes)
                for j in range(hidden_nodes)
            ],
            edge_color="green",
            width=1,
            style="dashed",
        )
        # 在边上添加权重
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={
                (
                    f"Hidden{num_layers}_{j + 1}",
                    f"Output{i + 1}",
                ): f"$w_{{{num_layers}{j + 1}{i + 1}}}$"
                for i in range(output_nodes)
                for j in range(hidden_nodes)
            },
            font_size=font_size,
            font_family="Times New Roman",
        )
        # ------------------------------------------------------------------------#
        plt.axis("off")
        # 标题：神经网络结构图英文
        plt.title(
            "Neural Networks Structure",
            fontdict={"size": 20},
            fontfamily="Times New Roman",
        )
        plt.show()


if __name__ == "__main__":
    # 创建实例
    GNN = Graph_Neural_Networks()
    # 设置属性
    GNN.input_nodes = 3
    GNN.hidden_nodes = 5
    GNN.num_layers = 6
    GNN.output_nodes = 4
    GNN.Graph_Neural_Networks(
        GNN.input_nodes, GNN.hidden_nodes, GNN.num_layers, GNN.output_nodes
    )
