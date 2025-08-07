import torch
import torch.nn.functional as F
import numpy as np

# 设置随机种子
torch.manual_seed(0)

# 1. 创建一组高斯分布的模拟“权重”
dim = 128
W = torch.randn(dim)
W[0] += 10
# 2. 构造一个随机正交（旋转）矩阵
def random_rotation_matrix(dim):
    Q, _ = torch.qr(torch.randn(dim, dim))  # QR分解得到正交矩阵
    return Q

R = random_rotation_matrix(dim)

print(f"ori weight:{W}, max:{torch.max(W)}, min:{torch.min(W)}, mean{torch.mean(W)}")
# 3. 应用旋转
W_rot = R @ W
print(f"rotated weight:{W_rot}, max:{torch.max(W_rot)}, min:{torch.min(W_rot)}, mean{torch.mean(W_rot)}")

# 4. 量化函数（对称8bit线性量化）
def quantize_symmetric(x, num_bits=8):
    qmin = -2**(num_bits-1)
    qmax = 2**(num_bits-1) - 1
    scale = x.abs().max() / qmax
    x_q = torch.clamp((x / scale).round(), qmin, qmax)
    x_deq = x_q * scale
    return x_deq, scale

# 5. 分别对原始和旋转权重量化
W_q, scale1 = quantize_symmetric(W)
W_rot_q, scale2 = quantize_symmetric(W_rot)

# 6. 反旋转回原始空间
# W_rot_q_inv = R.T @ W_rot_q

# 7. 计算 MSE
mse_original = F.mse_loss(W_q, W)
mse_rotated = F.mse_loss(W_rot_q, W)

# 8. 打印结果
print(f"Original Quantization MSE: {mse_original.item():.6f}")
print(f"Rotated Quantization MSE:  {mse_rotated.item():.6f}")
