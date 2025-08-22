import os
import torch
from flatquant.function_utils import get_paras_dict_by_name
import logging

def kronecker_matmul(x, hadL, hadR):
    """equivalent to
    
        had = torch.kron(hadL, hadR)
        x = x.reshape(-1, had.shape[0])
        x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)


def reparameterize_ln(ln, trans):
    # assert isinstance(ln, (LlamaRMSNorm, Qwen2RMSNorm))
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    ln_weight = ln_weight * trans.diag_scale.to(torch.float64)
    ln.weight.data = ln_weight.to(ori_dtype)
    trans.use_diag = False


def reparameterize_model(model):
    for idx in range(model.config.num_hidden_layers):
        layer = model.model.layers[idx]
        if hasattr(layer, 'self_attn') and hasattr(layer, 'mlp'):
            layer.self_attn.reparameterize()
            layer.mlp.reparameterize()
            if layer.self_attn.ln_trans is not None and layer.self_attn.ln_trans.add_diag:
                reparameterize_ln(layer.input_layernorm, layer.self_attn.ln_trans)
            if layer.mlp.up_gate_trans is not None and layer.mlp.up_gate_trans.add_diag:
                reparameterize_ln(layer.post_attention_layernorm, layer.mlp.up_gate_trans)
        elif hasattr(layer, 'attention') and hasattr(layer, 'feed_forward'):
            layer.attention.reparameterize()
            layer.feed_forward.reparameterize()
            if layer.attention.ln_trans is not None and layer.attention.ln_trans.add_diag:
                reparameterize_ln(layer.attention_norm, layer.attention.ln_trans)
            if layer.feed_forward.up_gate_trans is not None and layer.feed_forward.up_gate_trans.add_diag:
                reparameterize_ln(layer.ffn_norm, layer.feed_forward.up_gate_trans)
        # fuse per-channel scaling to layernorm

    return model


def save_parametrized_checkpoint(model, args):
    quanted_parameters = {}
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        quanted_parameters[i] = layer.state_dict()
    torch.save(quanted_parameters, os.path.join(args.exp_dir, f"parametrized_paras.pth"))
    logging.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"parametrized_paras.pth")))


def load_flat_parameters(args, model, path=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_parameters.pth"))
    else:
        # flat_parameters = torch.load(os.path.join(path, f"flat_parameters.pth"))
        flat_parameters = torch.load(path)
    layers = model.model.layers
    
    # pth 可以这么操作
    for i in range(len(flat_parameters.keys())):
        flat_param = flat_parameters[i]
        layers[i].load_state_dict(flat_param, strict=False)
    return model


def save_flat_matrices(args, model, rank=None):
    flat_matrices = {}
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        attn_block = layer.self_attn if hasattr(layer, 'self_attn') else layer.attention
        attn_block.rep_matrix_only()
        ff_block = layer.mlp if hasattr(layer, 'mlp') else layer.feed_forward
        ff_block.rep_matrix_only()
        paras_name = ["trans.matrix", "trans.diag_scale", "clip_factor_w", "clip_factor_a"]
        flat_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
    if rank is not None:
        matrices_path = os.path.join(args.exp_dir, f"flat_matrices_{rank}.pth")
    else:
        matrices_path = os.path.join(args.exp_dir, f"flat_matrices.pth")
    torch.save(flat_matrices, matrices_path)
    logging.info("saved paramaters at {}".format(matrices_path))


def load_flat_matrices(args, model, path=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_matrices.pth"))
    else:
        # flat_parameters = torch.load(os.path.join(path, f"flat_matrices.pth"))
        flat_parameters = torch.load(path)
    layers = model.model.layers
    
    for i in range(len(flat_parameters.keys())):
        flat_param = flat_parameters[i]
        # 主要将trans矩阵grad设为None
        layers[i].self_attn.rep_matrix_only()
        layers[i].mlp.rep_matrix_only()
        layers[i].load_state_dict(flat_param, strict=False)
    return model


