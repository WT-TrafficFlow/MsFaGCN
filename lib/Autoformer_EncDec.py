import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from lib.SelfAttention_Family import FullAttention


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class MovingAvg4D(nn.Module):
    """
    针对4D输入的移动平均模块，用于突出时间序列的趋势。
    输入 x 的形状: (batch_size, num_nodes, num_timesteps, num_features)
    在 num_timesteps 维度 (索引为2) 上进行平均。
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg4D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # 使用 unfold 和 mean 实现，不直接依赖 nn.AvgPool1d 以更好地控制维度

    def forward(self, x):
        # x: (B, N_nodes, T, F)
        B, N_nodes, T, F = x.shape

        # 在时间维度 (dim 2) 上进行填充
        # 左边（前端）填充的长度
        pad_front_len = (self.kernel_size - 1) - (self.kernel_size - 1) // 2
        # 右边（末端）填充的长度
        pad_end_len = (self.kernel_size - 1) // 2

        if pad_front_len < 0: pad_front_len = 0  # 确保不为负
        if pad_end_len < 0: pad_end_len = 0  # 确保不为负

        if T == 0:  # 处理空的时间序列输入
            # 如果原始时间序列长度为0，则无法进行填充和后续操作
            # 根据具体需求，可以返回空张量或特定形状的零张量
            # 这里我们计算预期的输出时间步数，如果可能的话
            if self.kernel_size > 0:
                # T_padded = T + pad_front_len + pad_end_len
                # T_out = math.floor((T_padded - self.kernel_size) / self.stride + 1)
                # 由于T=0, T_padded = kernel_size -1.
                # T_out = floor((kernel_size - 1 - kernel_size) / stride + 1) = floor(-1/stride + 1)
                # 如果 stride=1, T_out = 0. 如果 stride > 1 and kernel_size=1, T_out = 0.
                # 简单起见，如果输入T=0, 输出T_out也为0
                return torch.empty(B, N_nodes, 0, F, device=x.device, dtype=x.dtype)
            else:  # kernel_size is 0 or less, which is invalid for pooling
                return x

        # 前端填充：复制第一个时间步的数据
        # x[:, :, 0:1, :] 的形状是 (B, N_nodes, 1, F)
        front_slice = x[:, :, 0:1, :]
        front_padding = front_slice.repeat(1, 1, pad_front_len, 1)

        # 末端填充：复制最后一个时间步的数据
        # x[:, :, -1:, :] 的形状是 (B, N_nodes, 1, F)
        end_slice = x[:, :, -1:, :]
        end_padding = end_slice.repeat(1, 1, pad_end_len, 1)

        # 沿时间维度 (dim 2) 拼接
        x_padded = torch.cat([front_padding, x, end_padding], dim=2)
        # x_padded 形状: (B, N_nodes, T_padded, F)
        # T_padded = T + pad_front_len + pad_end_len = T + kernel_size - 1

        # 为了使用 unfold，将特征维度 F 和时间维度 T_padded 交换
        # 使得时间维度是最后一个维度，特征维度像是通道
        # x_permuted 形状: (B, N_nodes, F, T_padded)
        x_permuted = x_padded.permute(0, 1, 3, 2)

        # 沿时间维度 (现在是最后一个维度，索引为3) 进行 unfold
        # unfold(dimension, size, step)
        # dimension=3 (T_padded 维度)
        # size=self.kernel_size (窗口大小)
        # step=self.stride (步长)
        x_unfolded = x_permuted.unfold(3, self.kernel_size, self.stride)
        # x_unfolded 形状: (B, N_nodes, F, T_out, kernel_size)
        # T_out = floor((T_padded - kernel_size) / stride + 1)

        # 计算沿 kernel_size 维度 (最后一个维度) 的均值
        output_permuted = x_unfolded.mean(dim=-1)
        # output_permuted 形状: (B, N_nodes, F, T_out)

        # 将维度置换回原始顺序 (B, N_nodes, T_out, F)
        output = output_permuted.permute(0, 1, 3, 2)
        # output 形状: (B, N_nodes, T_out, F)

        return output

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        # self.moving_avg = moving_avg(kernel_size, stride=1)

        self.moving_avg = moving_avg(kernel_size=kernel_size, stride=1)
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(int(kernel), stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 


class FourierDecomp(nn.Module):
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        # res = x + y
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
