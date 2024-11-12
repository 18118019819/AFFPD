# 注意力,多头注意力,自注意力及Pytorch实现
# 使用了缩放点积作为打分函数，因此key和query的维数是一样的，实现很简单。
import logging

import torch
import numpy as np
from torch import nn


class ScaledDotProductAttension(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()  # 声明父类的Init方法
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)  # 沿哪一维实施softmax

    def forward(self, q, k, v, mask=None):
        # TORCH.BMM 执行batch内两矩阵乘积运算：bmm(b*n*m, b*m*p) -> size(b*n*p)
        # TORCH.BMM 输入必须是3-dim tensor
        u = torch.bmm(q, k.transpose(1, 2))  # matmul: matrix multiply
        u = u / self.scale  # 缩放

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)
        attn = self.softmax(u)
        output = torch.bmm(attn, v)
        return attn, output


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64
    batch = 32
    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)
    mask = torch.zeros(batch, n_q, n_k).bool()

    attension = ScaledDotProductAttension(scale=np.power(d_k, 0.5))
    attn, output = attension(q, k, v, mask=mask)
    # logging.info(f"attn:{attn}")
    # logging.info(f"output:{output}")
    print(f"q:{q.shape}")
    print(f"k:{k.shape}")
    print(f"v:{v.shape}")
    print(f"attn.size():{attn.size()}")
    print(f"output.size():{output.size()}")
# 开发者：ppy
# 时间：2024/4/17 15:00
