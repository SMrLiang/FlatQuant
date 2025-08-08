import torch
import torch.nn.functional as F

torch.manual_seed(0)

# --- helpers ---
def random_rotation(n):
    # Orthogonal via QR
    q, _ = torch.linalg.qr(torch.randn(n, n))
    # Ensure a proper rotation (det=+1); optional
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q

def quantize_symmetric(x, num_bits=8, axis=None):
    """
    Symmetric linear quant. If axis is not None, compute per-channel scales
    along that axis (kept for broadcasting).
    """
    qmin = -(2**(num_bits-1))
    qmax = (2**(num_bits-1)) - 1
    if axis is None:
        amax = x.abs().amax()
        scale = (amax / qmax) if amax > 0 else x.new_tensor(1.0)
    else:
        amax = x.abs().amax(dim=axis, keepdim=True)
        scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    x_q = torch.clamp((x / scale).round(), qmin, qmax)
    return x_q * scale, scale

# --- setup a linear layer and a batch of activations ---
B = 64          # batch
In = 128        # input dim
Out = 128       # output dim

W = torch.randn(Out, In)
X = torch.randn(B, In)

# sprinkle some outliers to make rotation matter
W[0, 0] += 10.0
X[0, 0] += 12.0

# Baseline float output
Y_fp = X @ W.T

# --- quantize without rotation ---
# per-tensor (set axis=None). For per-channel weight quant, use axis=1 (row-wise)
W_q, _ = quantize_symmetric(W, num_bits=8, axis=None)
X_q, _ = quantize_symmetric(X, num_bits=8, axis=None)
Y_wa_q = X_q @ W_q.T
mse_wa = F.mse_loss(Y_wa_q, Y_fp).item()

# --- apply rotations and quantize both weight & activation in rotated bases ---
U = random_rotation(Out)     # output rotation
V = random_rotation(In)      # input rotation

W_rot = U @ W @ V.T
X_rot = X @ V.T

# Quantize in rotated space (try per-channel weights along rows: axis=1)
W_rot_q, _ = quantize_symmetric(W_rot, num_bits=8, axis=1)
X_rot_q, _ = quantize_symmetric(X_rot, num_bits=8, axis=None)

# Compute output in rotated space then bring it back to original basis
Y_rot_q = X_rot_q @ W_rot_q.T           # this equals (approximately) Y_fp @ U.T
Y_rec = Y_rot_q @ U.T                   # back to original basis
mse_rot_wa = F.mse_loss(Y_rec, Y_fp).item()

print(f"MSE (quantize W & X, no rotation): {mse_wa:.6f}")
print(f"MSE (quantize W & X, with rotation): {mse_rot_wa:.6f}")
