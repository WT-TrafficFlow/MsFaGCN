import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import softmax
from typing import Optional
from torch_geometric.nn import MessagePassing
class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.2):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features // num_heads  # 每个头的输出维度
        self.num_heads = num_heads

        # 为每个头创建线性变换和注意力计算
        self.W = nn.ModuleList([nn.Linear(in_features, self.out_features) for _ in range(num_heads)])
        self.a = nn.ModuleList([nn.Linear(2 * self.out_features, 1) for _ in range(num_heads)])

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU()
        self.elu = nn.ELU()

    def forward(self, x, adj):
        head_outputs = []
        for head in range(self.num_heads):
            h = self.W[head](x)  # 对每个头应用线性变换

            # 计算注意力系数
            a_input = torch.cat([h.unsqueeze(2).expand(-1, -1, adj.size(0), -1, -1),
                                 h.unsqueeze(3).expand(-1, -1, -1, adj.size(1), -1)], dim=-1)
            e = self.leakyrelu(self.a[head](a_input)).squeeze(-1)  # (batch_size, num_nodes, num_nodes)

            # 掩码处理，防止在无连接处的注意力
            mask = adj == 0
            e.masked_fill_(mask, float('-inf'))

            # 计算注意力权重
            attention = F.softmax(e, dim=-1)
            attention = self.dropout(attention)

            # 应用注意力权重到节点特征
            h_prime = torch.matmul(attention, h)
            head_outputs.append(h_prime)

        # 将所有头的输出拼接
        h_out = torch.cat(head_outputs, dim=-1)  # (batch_size, num_nodes, num_heads*out_features)
        self.elu(h_out)
        return h_out

class GraphAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1):
        super(GraphAttentionTransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadGraphAttentionLayer(in_features, out_features, num_heads, dropout)
        self.ffn = nn.Linear(out_features, out_features)
        self.layer_norm1 = nn.LayerNorm(out_features)
        self.layer_norm2 = nn.LayerNorm(out_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, adj):
        # Multi-head graph attention
        att_out = self.attention(x, adj)
        att_out = self.dropout1(att_out)
        x = self.layer_norm1(x + att_out)

        # Feedforward network
        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self.layer_norm2(x + ffn_out)

        return x

class GraphAttentionTransformerEncoder(nn.Module):
    def __init__(self, num_layers, in_features, out_features, num_heads=4, dropout=0.1):
        super(GraphAttentionTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([GraphAttentionTransformerEncoderLayer(in_features, out_features, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x




class BERTSpaceTimeEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, num_nodes):
        super(BERTSpaceTimeEmbedding, self).__init__()
        self.time_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        self.space_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, input_ids):
        batch_size, sequence_length, num_nodes = input_ids.size(0), input_ids.size(-1), input_ids.size(-2)

        # Time embeddings
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)
        time_embeddings = self.time_embeddings(position_ids)

        # Space embeddings
        node_ids = torch.arange(num_nodes, dtype=torch.long, device=input_ids.device)
        node_ids = node_ids.unsqueeze(0).unsqueeze(1).expand(batch_size, sequence_length, num_nodes)
        space_embeddings = self.space_embeddings(node_ids)

        # Combine time and space embeddings
        space_time_embeddings = time_embeddings.unsqueeze(2) + space_embeddings

        return space_time_embeddings.transpose(1, 3)


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=3,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class GatedGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(GatedGraphConv, self).__init__()

        # Learnable weights for gating mechanism
        self.W = nn.Linear(in_channels, out_channels)
        self.U = nn.Linear(out_channels, out_channels)
        self.V = nn.Linear(out_channels, out_channels)

        # Attention mechanism
        self.attention = nn.Linear(2 * out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x, adj):
        """
        x: Tensor of shape (batch_size, num_nodes, in_channels)
        adj: Tensor of shape (num_nodes, num_nodes)  -> 2D adjacency matrix shared across batches
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation on input features
        h = self.W(x)  # Shape: (batch_size, num_nodes, out_channels)

        # Prepare for message passing
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (batch_size, num_nodes, num_nodes, out_channels)
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # (batch_size, num_nodes, num_nodes, out_channels)

        # Compute attention scores
        edge_features = torch.cat([h_i, h_j], dim=-1)  # (batch_size, num_nodes, num_nodes, 2 * out_channels)
        attention_weights = self.leakyrelu(self.attention(edge_features))  # (batch_size, num_nodes, num_nodes, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch_size, num_nodes, num_nodes)

        # Apply softmax to normalize the attention weights across edges
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Gate mechanism
        gate = torch.sigmoid(self.U(h_i) + self.V(h_j))  # (batch_size, num_nodes, num_nodes, out_channels)

        # Message passing: compute node updates
        node_features = gate * attention_weights.unsqueeze(-1) * h_j  # (batch_size, num_nodes, num_nodes, out_channels)

        # Apply adjacency matrix for message aggregation
        # Since adj is (num_nodes, num_nodes), we use batch matrix multiplication for each batch.
        aggr_out = torch.einsum("bijf,jk->bif", node_features, adj)  # Apply 2D adjacency matrix for aggregation

        # Apply layer normalization
        aggr_out = self.layer_norm(aggr_out)  # (batch_size, num_nodes, out_channels)

        return aggr_out


class GateGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.1):
        super(GateGCN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GatedGraphConv(in_channels, hidden_channels, dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GatedGraphConv(hidden_channels, hidden_channels, dropout))

        # Output layer
        self.layers.append(GatedGraphConv(hidden_channels, out_channels, dropout))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        """
        x: Tensor of shape (batch_size, num_nodes, in_channels)
        adj: Tensor of shape (num_nodes, num_nodes) -> Shared adjacency matrix for the whole batch
        """
        # Pass through Gated Graph Convolution layers
        for layer in self.layers[:-1]:
            x_res = x  # Residual connection
            x = layer(x, adj)
            x = self.relu(x + x_res)  # Residual connection with ReLU activation
            x = self.dropout(x)

        # Output layer without ReLU and dropout
        x = self.layers[-1](x, adj)
        return x


class GatedGraphConvPyG(MessagePassing):
    """
    基于 PyTorch Geometric 的门控图卷积层。
    这个层实现了注意力机制和门控机制，用于图节点特征的更新。
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        # aggr='add': 指定聚合方式为求和
        # node_dim=0: 指明节点特征在第0维 (num_nodes, features)
        super(GatedGraphConvPyG, self).__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 可学习的权重矩阵
        self.W = nn.Linear(in_channels, out_channels, bias=True)  # 节点特征变换
        self.U = nn.Linear(out_channels, out_channels, bias=True)  # 门控机制中的变换 (作用于目标节点 h_i)
        self.V = nn.Linear(out_channels, out_channels, bias=True)  # 门控机制中的变换 (作用于源节点 h_j)

        # 注意力机制
        self.attention_lin = nn.Linear(2 * out_channels, 1, bias=True)  # 计算注意力系数

        self.dropout_layer = nn.Dropout(dropout)  # Dropout 层
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)  # LeakyReLU激活函数
        self.layer_norm = nn.LayerNorm(out_channels)  # 层归一化

        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数"""
        self.W.reset_parameters()
        self.U.reset_parameters()
        self.V.reset_parameters()
        self.attention_lin.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        前向传播函数。
        Args:
            x (Tensor): 节点特征张量，形状为 (num_nodes, in_channels)。
            edge_index (Tensor): 边索引张量，形状为 (2, num_edges)，表示图的连接关系。
        Returns:
            Tensor: 更新后的节点特征张量，形状为 (num_nodes, out_channels)。
        """
        num_nodes_in_x = x.size(0)  # 假设 x 的形状是 [num_nodes, features]



        if edge_index.numel() > 0:  # 检查 edge_index 是否为空
            # 确保 edge_index 也在正确的设备上进行 min/max 操作
            min_val = edge_index.min().item()
            max_val = edge_index.max().item()


            if max_val >= num_nodes_in_x:
                print(
                    f"!!!!!! CRITICAL INDEXING ERROR in GatedGraphConvPyG: edge_index max value ({max_val}) is >= num_nodes_in_x ({num_nodes_in_x})")
                # 可以考虑保存问题数据
                # torch.save({'x': x.cpu(), 'edge_index': edge_index.cpu()}, 'gcn_critical_error_input.pt')
                raise IndexError(
                    f"Edge index max value {max_val} out of bounds for {num_nodes_in_x} nodes in GatedGraphConvPyG.")
            if min_val < 0:
                print(f"!!!!!! CRITICAL INDEXING ERROR in GatedGraphConvPyG: edge_index min value ({min_val}) is < 0")
                raise IndexError(f"Edge index min value {min_val} is negative in GatedGraphConvPyG.")




        num_nodes = x.size(0)

        # 1. 对输入节点特征进行线性变换
        # h_transformed 将用于在 message 方法中派生 h_i (目标节点特征) 和 h_j (源节点特征)
        h_transformed = self.W(x)  # 形状: (num_nodes, out_channels)

        # 2. 开始消息传递过程
        # propagate 方法会调用 message, aggregate (隐式通过aggr='add'), 和 update (如果定义)
        # 我们将 h_transformed 传递给 propagate，它会根据 edge_index 提取每条边的源节点和目标节点特征
        # size_i=num_nodes 用于确保 softmax 和聚合操作在正确的节点数量上进行
        aggr_out = self.propagate(edge_index, h=h_transformed, size_i=num_nodes)
        # aggr_out 形状: (num_nodes, out_channels)

        # 3. 应用层归一化
        aggr_out = self.layer_norm(aggr_out)

        return aggr_out

    def message(self, h_j: Tensor, h_i: Tensor, index: Tensor, ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        """
        构建消息的函数。对于每条边 (j -> i)，计算从 j 到 i 的消息。
        Args:
            h_j (Tensor): 源节点特征 (对于每条边)，形状 (num_edges, out_channels)。由 h[edge_index[1]] 得到。
            h_i (Tensor): 目标节点特征 (对于每条边)，形状 (num_edges, out_channels)。由 h[edge_index[0]] 得到。
            index (Tensor): 目标节点的索引 (edge_index[0])，形状 (num_edges)。用于 softmax。
            ptr (Optional[Tensor]): 用于批处理图的指针张量 (如果使用)。
            size_i (Optional[int]): 目标图中的节点数量 (即总节点数 num_nodes)。
        Returns:
            Tensor: 计算得到的消息，形状 (num_edges, out_channels)。
        """
        # 注意力机制
        # 拼接目标节点和源节点的特征
        edge_features = torch.cat([h_i, h_j], dim=-1)  # 形状: (num_edges, 2 * out_channels)
        # 计算原始注意力分数
        alpha = self.attention_lin(edge_features)  # 形状: (num_edges, 1)
        alpha = self.leakyrelu(alpha)
        # 使用 softmax 对注意力分数进行归一化 (针对每个目标节点的入边)
        alpha = softmax(src=alpha, index=index, ptr=ptr, num_nodes=size_i)  # 形状: (num_edges, 1)
        alpha = self.dropout_layer(alpha)  # 对归一化后的注意力分数应用 dropout

        # 门控机制
        # U 应用于目标节点特征 h_i, V 应用于源节点特征 h_j
        gate = torch.sigmoid(self.U(h_i) + self.V(h_j))  # 形状: (num_edges, out_channels)

        # 计算最终消息: 门控值 * 注意力加权的源节点特征
        messages = gate * alpha * h_j  # 形状: (num_edges, out_channels)
        return messages

    # update 方法在此处不是必需的，因为聚合后的结果直接在 forward 方法中进行了层归一化。
    # 如果需要在聚合结果和原始目标节点特征之间进行更复杂的操作 (例如 GRU 更新)，则需要实现 update 方法。


class GateGCNPyG(nn.Module):
    """
    使用 GatedGraphConvPyG 层堆叠的多层图神经网络模型。
    包含残差连接。
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int,
                 dropout: float = 0.1):
        super(GateGCNPyG, self).__init__()
        self.layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()  # 用于处理残差连接中维度不匹配的情况

        if num_layers == 1:
            # 如果只有一层，直接从 in_channels 映射到 out_channels
            self.layers.append(GatedGraphConvPyG(in_channels, out_channels, dropout))
            # self.skip_projections 在单层情况下不用于前向传播的残差计算
        else:
            # 输入层
            self.layers.append(GatedGraphConvPyG(in_channels, hidden_channels, dropout))
            if in_channels != hidden_channels:
                self.skip_projections.append(nn.Linear(in_channels, hidden_channels))
            else:
                self.skip_projections.append(nn.Identity())

            # 隐藏层
            for _ in range(num_layers - 2):
                self.layers.append(GatedGraphConvPyG(hidden_channels, hidden_channels, dropout))
                self.skip_projections.append(nn.Identity())  # 隐藏层之间维度相同

            # 输出层
            self.layers.append(GatedGraphConvPyG(hidden_channels, out_channels, dropout))
            # 输出层的残差连接不在此循环中处理，最后一层通常不带残差后的激活和dropout

        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        模型的前向传播。
        Args:
            x (Tensor): 初始节点特征，形状 (num_nodes, in_channels)。
            edge_index (Tensor): 边索引，形状 (2, num_edges)。
        Returns:
            Tensor: 最终的节点嵌入，形状 (num_nodes, out_channels)。
        """
        if not self.layers:  # 如果没有层 (例如 num_layers = 0)
            return x

        if len(self.layers) == 1:
            # 单层情况：直接通过该层，不加残差、ReLU和Dropout
            x = self.layers[0](x, edge_index)
            return x

        # 多层情况：应用残差连接、ReLU和Dropout (除了最后一层)
        # 迭代到倒数第二层（或所有参与残差连接的层）
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]

            # 获取对应的跳跃连接投影层 (如果需要)
            # 对于第一层之后且维度匹配的隐藏层，projection 是 nn.Identity()
            projection = self.skip_projections[i]

            x_res = projection(x)  # 计算残差项 (可能经过投影)

            x = layer(x, edge_index)  # 通过当前 GatedGraphConvPyG 层
            x = self.relu(x + x_res)  # 残差连接后跟ReLU激活
            x = self.dropout_layer(x)  # 应用Dropout

        # 最后一层：不应用ReLU和Dropout
        x = self.layers[-1](x, edge_index)
        return x
