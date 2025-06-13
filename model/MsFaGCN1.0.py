# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial
from lib.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp_multi
from lib.gat import GraphAttentionTransformerEncoder, MLP, GateGCN
from lib.FourierCorrelation import FourierBlock
from lib.AutoCorrelation import AutoCorrelationLayer
from lib.bi_mamba2 import BiMamba2_1D
from lib.Embed import BERTTimeEmbedding, PositionalEmbedding, FixedEmbedding, DoWEmbedding, TokenEmbedding
from lib.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from lib.Transformer_EncDec import Decoder
from torch import Tensor
from typing import Optional
from lib.mamba import Mamba, MambaConfig




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

        # 季节
        self.gat = GraphAttentionTransformerEncoder(1, self.enc_dim, self.enc_dim, self.num_heads, dropout=self.dropout)

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.enc_dim,
            nhead=self.num_heads,
            dim_feedforward=self.enc_dim * 4, batch_first=True)
        self.tat = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
        self.s_gate = FusionGate(self.enc_dim)

        # 趋势

        self.GateGCN = GateGCN(in_channels, in_channels, in_channels, 2)
        self.mamba = BI_Mamba(self.enc_dim)
        self.t_gate = FusionGate(self.enc_dim)
        self.ln = nn.LayerNorm(self.enc_dim)  #需要将channel放到最后一个维度上

    def forward(self, x, TE, SE, adj):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, node, num_of_timesteps, num_of_features = x.shape
        STE = TE + SE
        # 分解
        seasonal_init, trend_init = self.decomp(x.reshape(-1, num_of_timesteps, num_of_features))
        # 趋势
        trend_init = trend_init.reshape(-1, node, num_of_timesteps, num_of_features) + STE
        trend_init_t = trend_init
        trend_init_s = trend_init.transpose(1, 2).reshape(-1, node, num_of_features)

        trend_enc_s = self.GateGCN(trend_init_s, adj).reshape(-1, num_of_timesteps, node,
                                                                          num_of_features).transpose(1, 2)  # b n l f
        # trend_enc = self.cheb_conv_SAt(trend_init).transpose(2, 3)  # b n l f
        trend_enc_t = self.mamba(trend_init_t.reshape(-1, num_of_timesteps, num_of_features), bi_direction=True)
        trend_enc_t = trend_enc_t.reshape(-1, node, num_of_timesteps, num_of_features)  # b n l f
        trend_enc = self.t_gate(trend_enc_s, trend_enc_t)


        # 季节
        seasonal_init = seasonal_init.reshape(-1, node, num_of_timesteps, num_of_features)
        seasonal_init = seasonal_init + STE
        seasonal_init_s = seasonal_init.transpose(1, 2)
        seasonal_enc_s = self.gat(seasonal_init_s, adj).transpose(1, 2)  # b n l f
        # seasonal_enc, attn = self.tat(seasonal_enc.reshape(-1, num_of_timesteps, num_of_features))
        seasonal_init_t = seasonal_init.reshape(-1, num_of_timesteps, num_of_features)
        seasonal_enc_t = self.tat(seasonal_init_t)
        seasonal_enc_t = seasonal_enc_t.reshape(-1, node, num_of_timesteps, num_of_features)
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
        self.GateGCN = GateGCN(in_channels, in_channels, in_channels, 2)
        # 时间
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.enc_dim,
            nhead=self.num_heads,
            dim_feedforward=self.enc_dim * 4, batch_first=True)
        self.nat = nn.TransformerDecoder(self.TransformerDecoderLayer, num_layers=1)
        self.ln = nn.LayerNorm(self.enc_dim)  # 需要将channel放到最 后一个维度上

    def forward(self, x, x_enc, adj):

        batch_size, node, total_of_timesteps, num_of_features = x.shape
        x_dec = x.transpose(1, 2).reshape(-1, node, num_of_features)
        x_dec = self.GateGCN(x_dec, adj).reshape(-1, total_of_timesteps, node,
                                                                          num_of_features).transpose(1, 2)  # b n l f
        x_dec = x_dec.reshape(-1, total_of_timesteps, num_of_features)
        x_enc = x_enc.reshape(-1, self.seq_len, num_of_features)
        x_dec = self.nat(x_dec, x_enc).reshape(-1, node, total_of_timesteps, num_of_features)
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
        for block in self.enc_BlockList:
            x_e = block(x_e, te[..., :self.seq_len, :], pe[..., :self.seq_len, :], self.adj)
        x_d = F.pad(x_emb[..., :self.seq_len, :], (0, 0, 0, self.pred_len)) + pe + te
        for block in self.dec_BlockList:
            x_d = block(x_d, x_e, self.adj)

        output = self.final_proj(x_d[..., -self.pred_len:, :])

        output = output.squeeze()
        output = output * (self.std) + self.mean #(b,N,T)

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