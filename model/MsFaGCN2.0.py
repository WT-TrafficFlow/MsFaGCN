# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial
from lib.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp_multi, MovingAvg4D
from lib.gat import GraphAttentionTransformerEncoder, MLP, GateGCN, GateGCNPyG
from lib.FourierCorrelation import FourierBlock
from lib.AutoCorrelation import AutoCorrelationLayer
from lib.bi_mamba2 import BiMamba2_1D
from lib.Embed import BERTTimeEmbedding, PositionalEmbedding, FixedEmbedding, DoWEmbedding, TokenEmbedding
from lib.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from lib.Transformer_EncDec import Decoder
from torch import Tensor
from typing import Optional
from lib.mamba import Mamba, MambaConfig
import math

class TemporalMultiHeadAttention4D(nn.Module):
    """
    针对4D输入的时序多头注意力机制。
    在 num_timesteps 维度上进行注意力计算。
    可以用于自注意力 (query, key, value 相同) 或交叉注意力。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query: (B, N_nodes, T_q, d_model)
        # key:   (B, N_nodes, T_kv, d_model)
        # value: (B, N_nodes, T_kv, d_model)
        # B: batch_size, N_nodes: 节点数, T_q: 查询序列长度, T_kv: 键/值序列长度, d_model: 特征数

        B, N_nodes, T_q, _ = query.shape
        _, _, T_kv, _ = key.shape

        # 1. 线性投影
        q = self.wq(query)  # (B, N_nodes, T_q, d_model)
        k = self.wk(key)  # (B, N_nodes, T_kv, d_model)
        v = self.wv(value)  # (B, N_nodes, T_kv, d_model)

        # 2. 为多头重塑 Q, K, V
        # q: (B, N_nodes, num_heads, T_q, d_k)
        # k: (B, N_nodes, num_heads, T_kv, d_k)
        # v: (B, N_nodes, num_heads, T_kv, d_k)
        q = q.view(B, N_nodes, T_q, self.num_heads, self.d_k).permute(0, 1, 3, 2, 4)
        k = k.view(B, N_nodes, T_kv, self.num_heads, self.d_k).permute(0, 1, 3, 2, 4)
        v = v.view(B, N_nodes, T_kv, self.num_heads, self.d_k).permute(0, 1, 3, 2, 4)

        # 3. 计算注意力分数
        # attn_scores: (B, N_nodes, num_heads, T_q, T_kv)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask 的形状需要能广播到 attn_scores
            # 例如: (1, 1, 1, T_q, T_kv) 或 (B, N_nodes, 1, T_q, T_kv)
            # mask 中 0 (False) 代表需要被掩盖的位置
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 4. 注意力与 V 相乘
        # context: (B, N_nodes, num_heads, T_q, d_k)
        context = torch.matmul(attn_probs, v)

        # 5. 拼接多头结果
        # context: (B, N_nodes, T_q, d_model)
        context = context.permute(0, 1, 3, 2, 4).contiguous()
        context = context.view(B, N_nodes, T_q, self.d_model)

        output = self.fc(context)  # (B, N_nodes, T_q, d_model)
        return output


class PositionwiseFeedForward4D(nn.Module):
    """
    针对4D输入的逐点前馈网络。
    输入 x 的形状: (batch_size, num_nodes, num_timesteps, d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer4D(nn.Module):
    """
    单个Transformer编码器层，适配4D输入。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = TemporalMultiHeadAttention4D(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward4D(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: (B, N_nodes, T, d_model)

        # 自注意力 (query, key, value 都是 src)
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ffn_output = self.ffn(src)
        src = src + self.dropout2(ffn_output)
        src = self.norm2(src)

        return src


class TransformerEncoder4D(nn.Module):
    """
    完整的Transformer编码器。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer4D(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
        # self.norm = nn.LayerNorm(d_model) # 可选的最终层归一化

    def forward(self, src, src_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)
        # if self.norm is not None:
        #     output = self.norm(output)
        return output


class TransformerDecoderLayer4D(nn.Module):
    """
    单个Transformer解码器层，适配4D输入。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 掩码自注意力 (对目标序列)
        self.self_attn = TemporalMultiHeadAttention4D(d_model, num_heads, dropout)
        # 交叉注意力 (Q from target, K/V from encoder memory)
        self.cross_attn = TemporalMultiHeadAttention4D(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward4D(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后
        self.norm3 = nn.LayerNorm(d_model)  # FFN后

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: (B, N_nodes, T_tgt, d_model) - 解码器输入 (目标序列)
        # memory: (B, N_nodes, T_src, d_model) - 编码器输出
        # tgt_mask: 目标序列的掩码 (例如，因果掩码)
        # memory_mask: 编码器输出的掩码 (例如，源序列的padding掩码)

        # 1. 掩码自注意力 (Masked Self-Attention)
        # query, key, value 都是 tgt
        self_attn_output = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(self_attn_output)
        tgt = self.norm1(tgt)

        # 2. 交叉注意力 (Encoder-Decoder Attention)
        # query 是 tgt, key 和 value 是 memory
        cross_attn_output = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = tgt + self.dropout2(cross_attn_output)
        tgt = self.norm2(tgt)

        # 3. 前馈网络 (Feed-Forward Network)
        ffn_output = self.ffn(tgt)
        tgt = tgt + self.dropout3(ffn_output)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder4D(nn.Module):
    """
    完整的Transformer解码器。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer4D(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)  # 最终的层归一化

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: (B, N_nodes, T_tgt, d_model)
        # memory: (B, N_nodes, T_src, d_model)
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        output = self.norm(output)  # 应用最终的层归一化
        return output


class PositionwiseFeedForward4D(nn.Module):
    """
    针对4D输入的逐点前馈网络。
    输入 x 的形状: (batch_size, num_nodes, num_timesteps, d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一个线性层，扩展维度
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二个线性层，恢复维度
        self.activation = nn.ReLU()  # 或者使用 nn.GELU()

    def forward(self, x):
        # x: (B, N_nodes, T, d_model)
        # 对最后一个维度 (d_model) 进行操作
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer4D(nn.Module):
    """
    单个Transformer编码器层，适配4D输入。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = TemporalMultiHeadAttention4D(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward4D(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)  # 在注意力之后
        self.norm2 = nn.LayerNorm(d_model)  # 在前馈网络之后

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: (B, N_nodes, T, d_model)

        # 1. 自注意力模块 + Add & Norm
        attn_output = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)  # 残差连接
        src = self.norm1(src)  # 层归一化

        # 2. 前馈网络模块 + Add & Norm
        ffn_output = self.ffn(src)
        src = src + self.dropout2(ffn_output)  # 残差连接
        src = self.norm2(src)  # 层归一化

        return src


class TransformerEncoder4D(nn.Module):
    """
    完整的Transformer编码器，由多个编码器层堆叠而成。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer4D(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
        # 根据需要，可以在所有层之后再加一个 LayerNorm
        # self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # src: (B, N_nodes, T, d_model)
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)

        # if self.norm is not None:
        #     output = self.norm(output)
        return output


class BI_Mamba(nn.Module):
    def __init__(self, in_channels):
        super(BI_Mamba, self).__init__()
        self.mamba_Config = MambaConfig(d_model=in_channels, n_layers=1)
        self.mamba_for = Mamba(self.mamba_Config)
        self.mamba_back = Mamba(self.mamba_Config)
        self.proj = nn.Linear(in_channels*2, in_channels)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(in_channels)  #需要将channel放到最后一个维度上

    def forward(self, x, bi_direction=True):
        if bi_direction:
            x1 = self.mamba_for(x)
            x2 = self.mamba_back(x.flip(1)).flip(1)
            y = torch.cat([x1, x2], dim=-1)
            y = self.relu(self.proj(y))
            # y = self.proj(x1, x2)
            x = self.ln(x+y)
        else:
            x = self.mamba2_for(x)
        return x


class FusionGate(nn.Module):
    def __init__(self, input_dim):
        super(FusionGate, self).__init__()
        # 用于生成gate的线性层
        self.gate = nn.Linear(input_dim * 2, input_dim)  # 假设两个输入维度相同，乘以2表示拼接后
        self.sigmoid = nn.Sigmoid()  # 用于将gate值限制在(0, 1)范围内
        self.relu = nn.ReLU()

    def forward(self, input1, input2):
        # 假设input1和input2的维度相同：[batch_size, input_dim]

        # 拼接两个输入 [batch_size, input_dim * 2]
        combined_input = torch.cat([input1, input2], dim=-1)

        # 通过线性层生成gate值，大小为[batch_size, 1]
        gate_value = self.relu(self.gate(combined_input))

        # 对两个输入进行加权 [batch_size, input_dim]
        # output = gate_value * input1 + (1 - gate_value) * input2
        output = gate_value
        return output


class CustomTransformerDecoderLayerNoSelfAttention(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first: bool = False):
        super(CustomTransformerDecoderLayerNoSelfAttention, self).__init__(d_model, nhead, dim_feedforward, dropout,
                                                                           activation, batch_first=batch_first)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer, excluding self-attention.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
            memory_is_causal: If specified, applies a causal mask as ``memory mask``.

        Shape:
            see the docs in Transformer class.
        """

        x = tgt
        if self.norm_first:
            # Removed self-attention (_sa_block)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            # Removed self-attention (_sa_block)
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class MAMGCN_encblock(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, num_of_timesteps, kernel_size):
        super(MAMGCN_encblock, self).__init__()
        self.enc_dim = in_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = num_of_timesteps
        # 分解

        self.decomp = series_decomp_multi(kernel_size)

        self.gat = GraphAttentionTransformerEncoder(1, self.enc_dim, self.enc_dim, self.num_heads, dropout=self.dropout)

        self.tat = TransformerEncoder4D(
            num_layers=1,
            d_model=self.enc_dim,
            num_heads=self.num_heads,
            d_ff=self.enc_dim * 4,
            dropout=self.dropout
        )
        self.s_gate = FusionGate(self.enc_dim)





        # 趋势
        self.GateGCN = GateGCNPyG(in_channels, in_channels, in_channels, 2, self.dropout)
        # self.GateGCN = GateGCN(in_channels, in_channels, in_channels, 2)
        self.mamba = BI_Mamba(self.enc_dim)
        self.t_gate = FusionGate(self.enc_dim)
        self.ln = nn.LayerNorm(self.enc_dim)  #需要将channel放到最后一个维度上

    def forward(self, x, TE, SE, adj, edge_index):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, node, num_of_timesteps, num_of_features = x.shape
        STE = TE + SE
        # 分解
        # 分解
        seasonal_init, trend_init = self.decomp(x.reshape(-1, num_of_timesteps, num_of_features))
        # 趋势
        trend_init_t = trend_init.reshape(-1, node, num_of_timesteps, num_of_features) + STE
        # seasonal_init, trend_init = self.decomp(x) # batch_size, node, num_of_timesteps, num_of_features
        # # 趋势
        # trend_init = trend_init + STE
        trend_init_s = trend_init_t.transpose(1, 2).reshape(-1, node, num_of_features)
        B_eff = trend_init_s.size(0)
        x_for_gcn = trend_init_s.reshape(-1, num_of_features)
        edge_indices_list = []
        for i in range(B_eff):
            # 将当前图的节点索引加上偏移量 i * N
            edge_indices_list.append(edge_index + i * node)
        # 将所有调整后的 edge_index 连接起来
        batched_edge_index = torch.cat(edge_indices_list, dim=1).to(x_for_gcn.device)  # 确保设备一致
        trend_enc_s = self.GateGCN(x_for_gcn, batched_edge_index).reshape(-1, num_of_timesteps, node, num_of_features).transpose(1, 2)  # b n l f
        trend_enc_t = self.mamba(trend_init_t.reshape(-1, num_of_timesteps, num_of_features), bi_direction=True)
        trend_enc_t = trend_enc_t.reshape(-1, node, num_of_timesteps, num_of_features)  # b n l f
        trend_enc = self.t_gate(trend_enc_s, trend_enc_t)


        # 季节
        seasonal_init = seasonal_init.reshape(-1, node, num_of_timesteps, num_of_features) + STE
        seasonal_init_s = seasonal_init.transpose(1, 2)
        seasonal_enc_s = self.gat(seasonal_init_s, adj).transpose(1, 2)  # b n l f
        seasonal_enc_t = self.tat(seasonal_init)
        seasonal_enc = self.s_gate(seasonal_enc_s, seasonal_enc_t)



        x_residual = x + seasonal_enc + trend_enc
        return x_residual



class MAMGCN_decblock(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(MAMGCN_decblock, self).__init__()
        self.enc_dim = in_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = num_of_timesteps
        # 空间
        self.GateGCN = GateGCNPyG(in_channels, in_channels, in_channels, 2, self.dropout)
        self.nat = TransformerDecoder4D(
            num_layers=1,
            d_model=self.enc_dim,
            num_heads=num_heads,
            d_ff=self.enc_dim * 4,
            dropout=self.dropout
        )
        self.ln = nn.LayerNorm(self.enc_dim)  # 需要将channel放到最 后一个维度上

    def forward(self, x, x_enc, adj, edge_index):

        batch_size, node, total_of_timesteps, num_of_features = x.shape
        x_dec = x.transpose(1, 2).reshape(-1, node, num_of_features)
        B_eff = x_dec.size(0)
        x_for_gcn = x_dec.reshape(-1, num_of_features)
        edge_indices_list = []
        for i in range(B_eff):
            # 将当前图的节点索引加上偏移量 i * N
            edge_indices_list.append(edge_index + i * node)
        # 将所有调整后的 edge_index 连接起来
        batched_edge_index = torch.cat(edge_indices_list, dim=1).to(x_for_gcn.device)  # 确保设备一致
        x_dec = self.GateGCN(x_for_gcn, batched_edge_index).reshape(-1, total_of_timesteps, node,
                                                                          num_of_features).transpose(1, 2)  # b n l f
        x_dec = self.nat(x_dec, x_enc)
        x_dec = x + x_dec

        return x_dec





class MAMGCN_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_for_predict, len_input, num_of_vertices, mean, std,
                 kernel_size=None, adj=None):


        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(MAMGCN_submodule, self).__init__()
        if kernel_size is None:
            kernel_size = [2, 3]
        self.mean = mean
        self.std = std
        self.adj = torch.from_numpy(adj).float().to(DEVICE)
        self.edge_index = adj_to_edge_index(self.adj).detach()
        # self.emb = nn.Linear(1, in_channels)
        self.emb = MLP(
            1, in_channels,
            hidden_dim=in_channels // 2,
            hidden_layers=2,
            dropout=dropout,
            activation='relu')
        self.enc_dim = in_channels
        self.relu = nn.ReLU()
        self.node = num_of_vertices
        self.ve = TokenEmbedding(in_channels, in_channels)
        self.pe = BERTTimeEmbedding(max_position_embeddings=num_of_vertices, embedding_dim=in_channels)
        self.tod = MLP(
            1, in_channels,
            hidden_dim=in_channels // 2,
            hidden_layers=2,
            dropout=dropout,
            activation='relu')
        self.dow = DoWEmbedding(in_channels)
        self.seq_len = len_input
        self.pred_len = num_for_predict
        self.enc_BlockList = nn.ModuleList([MAMGCN_encblock(DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, len_input, kernel_size) for _ in range(nb_block)])
        self.dec_BlockList = nn.ModuleList([MAMGCN_decblock(DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, len_input)
                                            for _ in range(nb_block)])

        self.final_proj = MLP(
                 in_channels,
                 1,
                 hidden_dim=in_channels//2,
                 hidden_layers=2,
                 dropout=dropout,
                 activation='relu')
        # self.final_proj = nn.Linear(in_channels, 1)

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        global x_enc, x_dec
        x_flow = x[..., :1]
        x_emb = self.emb(x_flow)
        x_e = x_emb[..., :self.seq_len, :]
        tod = self.tod(x[..., 1:2])
        dow = self.dow(x[..., 2])
        te = tod + dow
        pe = self.relu(self.pe(x_emb))
        x_e = x_e + te[..., :self.seq_len, :] + pe[..., :self.seq_len, :]
        for block in self.enc_BlockList:
            x_e = block(x_e, te[..., :self.seq_len, :], pe[..., :self.seq_len, :], self.adj, self.edge_index)
        x_d = F.pad(x_emb[..., :self.seq_len, :], (0, 0, 0, self.pred_len)) + pe + te
        for block in self.dec_BlockList:
            x_d = block(x_d, x_e, self.adj, self.edge_index)

        output = self.final_proj(x_d[..., -self.pred_len:, :])
        output = output.squeeze()
        output = output * (self.std) + self.mean #(b,N,T)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, num_heads, dropout, adj_mx, num_for_predict, len_input, num_of_vertices, mean, std, kernel_size):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = MAMGCN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_for_predict,
         len_input, num_of_vertices, mean, std, kernel_size, adj_mx)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model

def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """
    将一个 N*N 的邻接矩阵 (可以是稠密或稀疏的 PyTorch 张量)
    转换为 PyTorch Geometric 格式的 edge_index。

    Args:
        adj (torch.Tensor): N*N 的邻接矩阵。
                            - 如果是稠密张量, 非零元素表示边。
                            - 如果是稀疏张量 (例如 torch.sparse_coo_tensor),
                              其存储的非零元素定义了边。
                            数据类型应为 float32 (或任何数值类型)，但函数主要关心结构。

    Returns:
        torch.Tensor: 形状为 (2, num_edges) 的 edge_index 张量，数据类型为 torch.long。
                      edge_index[0] 是源节点索引列表。
                      edge_index[1] 是目标节点索引列表。
    """
    if not isinstance(adj, torch.Tensor):
        raise TypeError(f"输入必须是 PyTorch 张量，但收到了 {type(adj)}")

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"邻接矩阵必须是 N*N 的二维方阵, 但收到了形状为 {adj.shape} 的张量"
        )

    if adj.is_sparse:
        # 如果是稀疏张量 (例如 torch.sparse_coo_tensor, torch.sparse_csr_tensor 等)
        # PyTorch Geometric 通常使用 COO 格式的索引
        if adj.layout == torch.sparse_coo:
            edge_index = adj.coalesce()._indices()
        elif adj.layout == torch.sparse_csr or adj.layout == torch.sparse_csc:
            # 对于 CSR/CSC, 先转换为 COO
            edge_index = adj.to_sparse_coo().coalesce()._indices()
        else:
            raise ValueError(f"不支持的稀疏张量格式: {adj.layout}. 请使用 COO, CSR, 或 CSC。")
    else:
        # 如果是稠密张量
        if torch.isnan(adj).any():
            # torch.save(adj, "debug_adj_with_nan.pt") # 可选：保存张量以供调试
            raise ValueError("Input adjacency matrix (dense) to adj_to_edge_index contains NaNs!")
        if torch.isinf(adj).any():
            # torch.save(adj, "debug_adj_with_inf.pt") # 可选：保存张量以供调试
            raise ValueError("Input adjacency matrix (dense) to adj_to_edge_index contains Infs!")

        edge_index = torch.nonzero(adj, as_tuple=False).t()


    # 确保 edge_index 的数据类型是 torch.long
    return edge_index.long()