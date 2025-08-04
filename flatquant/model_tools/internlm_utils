import math

import torch
import torch.nn as nn

from flatquant.quant_utils import ActivationQuantizer
from flatquant.utils import skip_initialization
from flatquant.function_utils import get_init_scale, get_decompose_dim
from flatquant.trans_utils import SVDSingleTransMatrix, SVDDecomposeTransMatrix
from flatquant.trans_utils import InvSingleTransMatrix, InvDecomposeTransMatrix
from flatquant.flat_linear import FlatQuantizedLinear
from einops import rearrange
import sys
BASE = "/home/liangyiheng/xten/models/internlm/internlm2_1_8b"
sys.path.insert(0, BASE)

from hf.modeling_internlm2 import InternLM2Attention, InternLM2MLP, repeat_kv, apply_rotary_pos_emb

class FlatQuantInternLM2MLP(InternLM2MLP):
    def __init__(self, args, module: InternLM2MLP):
        self.args = args
        self.w1 = FlatQuantizedLinear(args, module.w1)
        self.w3 = FlatQuantizedLinear(args, module.w3)
        self.w2 = FlatQuantizedLinear(args, module.w2)
        self.add_fq_trans()
        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            # 不同0维取最大
            self.up_smax = torch.ones_like(self.w1.linear.weight.abs().max(dim=0)[0]).cuda() * 1e-5
            self.down_smax = torch.ones_like(self.w2.linear.weight.abs().max(dim=0)[0]).cuda() * 1e-5

    
    def add_fq_trans(self):
        if self.args.direct_inv:
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            DecomposeTransMatrix = SVDDecomposeTransMatrix

        if self.args.w_bits < 16 or self.args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.w1.linear.weight.shape[1])
            self.up_gate_trans = DecomposeTransMatrix(up_dim_left, up_dim_right, add_diag=self.args.add_diag)
            down_dim_left, down_dim_right = get_decompose_dim(self.w2.linear.weight.shape[1])
            self.down_trans = DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=self.args.add_diag)
        else:
            self.up_gate_trans, self.down_trans = None, None        


    def _trans_forward(self, x):
        if self.up_gate_trans is not None:
            x_ts = self.up_gate_trans(x)
        else:
            x_ts = x
        up_states = self.w1(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.w3(x_ts, qa_trans=self.up_gate_trans)

        x_act_fn = self.act_fn(gate_states) * up_states
        if self.down_trans is not None:
            x_ts_2 = self.down_trans(x_act_fn)
        else:
            x_ts_2 = x_act_fn
        down_states = self.w2(x_ts_2, qa_trans=self.down_trans)
        return down_states



    def _ori_forward(self, x):
        '''origin implement: w2 = self.w2(self.act_fn(self.w3(x)) * self.w1(x))'''
        if self.diag_init == "sq_style":
            self.up_smax = torch.maximum(self.up_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        x = self.act_fn(self.w3._ori_forward(x)) * self.w1._ori_forward(x)
        if self.diag_init == "sq_style":
            self.down_smax = torch.maximum(self.down_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        down_states = self.w2._ori_forward(x)
        return down_states


    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)



class FlatQuantInternLM2Attention(InternLM2Attention):
    def __init__(self, args, module: InternLM2Attention):
        super().__init__(module.config, module.layer_idx)
        self.args = args
        self.config = module.config 
        # self.q_proj = FlatQuantizedLinear(args, module.q_proj)
        # self.k_proj = FlatQuantizedLinear(args, module.k_proj)
        # self.v_proj = FlatQuantizedLinear(args, module.v_proj)
        self.wqkv = FlatQuantizedLinear(args, module.wqkv)

        self.wo = FlatQuantizedLinear(args, module.wo)
        self.add_fq_trans()
        
        # get num_key_value_groups
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads        
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads


        assert args.q_bits == args.k_bits == args.v_bits

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(bits=args.q_bits, \
                                        sym=not(args.q_asym), lac=args.lac, groupsize=-1, )
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(bits=args.k_bits, \
                                        sym=not(args.k_asym), lac=args.lac, groupsize=-1, )
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(bits=args.v_bits, \
                                        sym=not(args.v_asym), lac=args.lac, groupsize=-1, )

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            self.ln_smax = torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0]).cuda() * 1e-5

    def add_fq_trans(self):
        if self.args.direct_inv:
            SingleTransMatrix, DecomposeTransMatrix = InvSingleTransMatrix, InvDecomposeTransMatrix
        else:
            SingleTransMatrix, DecomposeTransMatrix = SVDSingleTransMatrix, SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(self.q_proj.linear.weight.shape[1])
            self.ln_trans = DecomposeTransMatrix(ln_dim_left, ln_dim_right, add_diag=self.args.add_diag)
            self.o_trans = SingleTransMatrix(self.config.num_attention_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = SingleTransMatrix(head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = SingleTransMatrix(head_dim)
        else:
            self.vcache_trans = None

    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        # query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        # key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        # if self.args.separate_vtrans:
        #     value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        # else:
        #     value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        
        qkv_states = self.wqkv(hidden_states, qa_trans=self.ln_trans)
        return qkv_states

    def _ori_forward_after_ln(self, hidden_states):
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(self.ln_smax, \
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach())
        # query_states = self.q_proj._ori_forward(hidden_states)
        # key_states = self.k_proj._ori_forward(hidden_states)
        # value_states = self.v_proj._ori_forward(hidden_states)
        qkv_states = self.wqkv._ori_forward(hidden_states)
        return qkv_states

    def quant_vcache(self, value_states):
        if self.args.separate_vtrans:
            value_states = self.vcache_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states

    def quant_kcache(self, q, k):
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        # Q/K transform
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q)
        # TODO: by default do the per-head quantizaion for k-v-cache
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k

    def forward(self, hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache, **kwargs):
        # all forward based on pretraining_tp=1
        assert self.config.pretraining_tp == 1

        bsz, q_len, _ = hidden_states.size()
        if self._ori_mode:
            qkv_states = self._ori_forward_after_ln(hidden_states)
        else:
            qkv_states = self._trans_forward_after_ln(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]
        
        assert query_states[-1] == key_states[-1] == self.num_key_value_heads == value_states[-1]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # ---- here do the quantization ----
        # hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhere to continue
        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups) # bnsh
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : kv_seq_len]
            if causal_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {causal_mask.size()}"
                )
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        

        

        if self._ori_mode:
            attn_output = self.wo._ori_forward(attn_output)
        else:
            # new foward: 
            if self.o_trans is None and self.vcache_trans is not None:
                # attn_output = self.vcache_trans(value_states)
                init_shape = attn_output.shape
                attn_output = attn_output.reshape(-1, self.config.num_attention_heads, self.config.hidden_size//self.config.num_attention_heads)
                # .to设备和数据类型都保持一致
                attn_output = torch.matmul(attn_output, self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output)).reshape(init_shape)
                attn_output = self.wo(attn_output)
            else:
                init_shape = attn_output.shape
                attn_output = attn_output.reshape(-1, self.config.num_attention_heads, self.config.hidden_size//self.config.num_attention_heads)
                attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
                    attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
                    attn_output = self.wo(attn_output, qa_trans=[attn_o_og_it, attn_v_og_it])
                else:
                    attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        # self.q_proj.reparameterize(qa_trans=self.ln_trans)
        # self.k_proj.reparameterize(qa_trans=self.ln_trans)
        # if self.args.separate_vtrans:
        #     self.v_proj.reparameterize(qa_trans=self.ln_trans)
        # else:
        #     self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)

        self.wqkv.reparameterize(qa_trans=self.ln_trans)
        
        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            self.wo.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        # qkvw_smax = torch.cat([self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        
        qkvw_smax = self.wqkv.linear.weight.abs().max(dim=0)[0]


        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax, alpha)
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(self, ):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


def apply_flatquant_to_llama(args, model):
    skip_initialization()
    # Replace module with FlatQuant version
    for layer in range(model.config.num_hidden_layers):
        # attn
        model.model.layers[layer].self_attn = FlatQuantInternLM2Attention(args, model.model.layers[layer].self_attn)
        # mlp
        model.model.layers[layer].mlp = FlatQuantInternLM2MLP(args, model.model.layers[layer].mlp)
    return model
