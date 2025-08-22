import torch
from tqdm import tqdm
import torch.nn.functional as F
# batch size = 1
@torch.no_grad()
def ppl_eval(model, testenc):
    print('Evaluating ppl...')
    model.eval()
    max_length = 2048   # fix model max length

    testenc = testenc.input_ids
    nsamples = testenc.numel() // max_length

    dev = next(model.parameters()).device

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * max_length): ((i + 1) * max_length)]
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * max_length): ((i + 1) * max_length)
        ][:, 1:].to(shift_logits.device)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * max_length
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * max_length))
    return ppl.item()

def extract_key_values(pkv):
    """Return lists [K_l], [V_l] from various PKV formats."""
    # New API: Cache object with key_cache/value_cache
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        Ks = list(pkv.key_cache)
        Vs = list(pkv.value_cache)
        return Ks, Vs

    # Single layer (K, V)
    if isinstance(pkv, tuple) and len(pkv) == 2 and torch.is_tensor(pkv[0]) and torch.is_tensor(pkv[1]):
        return [pkv[0]], [pkv[1]]

    # Sequence of layers: list/tuple of (K,V)
    if isinstance(pkv, (list, tuple)) and len(pkv) > 0 and isinstance(pkv[0], (list, tuple)) and len(pkv[0]) >= 2:
        # 原版是这种
        Ks = [kv[0] for kv in pkv]
        Vs = [kv[1] for kv in pkv]
        return Ks, Vs

    raise TypeError(f"Unsupported past_key_values type: {type(pkv)}")


def mse_between_pkv(pkv_a, pkv_b):
    """Compute mean MSE across layers for K and V."""
    Ka, Va = extract_key_values(pkv_a)
    Kb, Vb = extract_key_values(pkv_b)
    assert len(Ka) == len(Kb) == len(Va) == len(Vb), "Layer count mismatch"

    mse_k_total, mse_v_total = 0.0, 0.0
    num_layers = len(Ka)

    for layer_idx in range(num_layers):
        k_a, v_a = (Ka[layer_idx]).to(torch.float32),(Va[layer_idx]).to(torch.float32)
        k_b, v_b = (Kb[layer_idx]).to(torch.float32), (Vb[layer_idx]).to(torch.float32)

        # If your cache uses [T,B,H,D] instead of [B,H,T,D], permute here:
        # k_a = k_a.permute(1,2,0,3); v_a = v_a.permute(1,2,0,3)
        # k_b = k_b.permute(1,2,0,3); v_b = v_b.permute(1,2,0,3)

        # Align seq_len to the minimum (decode steps may differ by 1)
        Ta, Tb = k_a.shape[-2], k_b.shape[-2]
        assert Ta == Tb
        T = min(Ta, Tb)
        k_a = k_a[..., :T, :]
        k_b = k_b[..., :T, :]
        v_a = v_a[..., :T, :]
        v_b = v_b[..., :T, :]

        mse_k_total += F.mse_loss(k_a, k_b, reduction="mean").item()
        mse_v_total += F.mse_loss(v_a, v_b, reduction="mean").item()

    return mse_k_total / num_layers, mse_v_total / num_layers


@torch.no_grad()
def mse_eval_internlm(ori_model, model, testenc, tokenizer):
    ori_model.cuda()
    model.eval()
    ori_model.eval()
    max_length = 2048   # fix model max length
    testenc = testenc.input_ids

    nsamples = testenc.numel() // max_length

    dev = next(model.parameters()).device

    testenc = testenc.to(dev)    
    prefill_acc = []
    prefill_mse = []
    decode_acc = []
    seq_len = 541   

    mse_fn = torch.nn.MSELoss(reduction='mean')
    hidden_mse = []
    # --------------prefill to first token---------------
    for i in tqdm(range(nsamples)):
        # 右侧是开区间
        batch = testenc[:, (i * max_length): (i * max_length + seq_len)]
        with torch.no_grad():
            lm_logits = ori_model(batch).logits
            qlm_logits = model(batch).logits   
        pred = lm_logits.argmax(dim=-1)        # (bsz, seq_len)
        q_pred = qlm_logits.argmax(dim=-1)
        loss = mse_fn(lm_logits, qlm_logits)
        prefill_mse.append(loss.item())
        prefill_acc.append((pred == q_pred).sum().item())
        loss = mse_fn(lm_logits[:,-1,:], qlm_logits[:,-1,:])
        hidden_mse.append(loss)

        del lm_logits, qlm_logits, loss
        torch.cuda.empty_cache()     
    prefill_acc = sum(prefill_acc) / (nsamples * seq_len)
    prefill_mse = sum(prefill_mse) / len(prefill_mse)
    hidden_mse = sum(hidden_mse) / len(hidden_mse)

    # ------------several tokens------------------
    new_tokens = 8
    pad_token_id = tokenizer.pad_token_id
    k_mse = []
    v_mse = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * max_length): (i * max_length + seq_len)]
        attn_mask = (batch != pad_token_id).long()
        out = ori_model(batch, use_cache=True)
        q_out = model(batch, use_cache=True)
        past_key_values = out.past_key_values
        q_past_key_values = q_out.past_key_values

        mse_k, mse_v = mse_between_pkv(past_key_values, q_past_key_values)
        k_mse.append(mse_k)
        v_mse.append(mse_v)

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
        q_next_token = torch.argmax(q_out.logits[:, -1, :], dim=-1)
        correct = 0 
        if next_token != q_next_token:
            continue 
        correct += 1
        ones = torch.ones(1, 1, dtype=attn_mask.dtype, device=dev)
        attn_mask = torch.cat([attn_mask, ones], dim=1) 
          
        next_token = next_token.unsqueeze(-1)
        q_next_token = q_next_token.unsqueeze(-1)        

        for i in range(new_tokens-1):                            
            pos_id = attn_mask.long().cumsum(-1) - 1
            pos_id.masked_fill_(attn_mask == 0, 1)
            pos_id = pos_id[:, -1 :]
            q_out = model(input_ids = q_next_token, attention_mask=attn_mask.clone(), position_ids=pos_id, use_cache=True, past_key_values=q_past_key_values)
            out = ori_model(input_ids = next_token, attention_mask=attn_mask.clone(), position_ids=pos_id, use_cache=True, past_key_values=past_key_values)
            loss = mse_fn(q_out.logits, out.logits)
            
            past_key_values = out.past_key_values
            q_past_key_values = q_out.past_key_values
            
            if next_token != q_next_token:
                break
            correct += 1
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
            q_next_token = torch.argmax(q_out.logits[:, -1, :], dim=-1)

            next_token = next_token.unsqueeze(-1)
            q_next_token = q_next_token.unsqueeze(-1)

            ones = torch.ones(1, 1, dtype=attn_mask.dtype, device=dev)
            attn_mask = torch.cat([attn_mask, ones], dim=1)

        decode_acc.append(correct / new_tokens)            

    decode_acc = sum(decode_acc) / (nsamples * new_tokens)
    k_mse = sum(k_mse) / len(k_mse)
    v_mse = sum(v_mse) / len(v_mse)

    ori_model.cpu()
    return prefill_acc, decode_acc, hidden_mse, k_mse, v_mse


@torch.no_grad()
def mse_eval_qwen(ori_model, model, testenc, tokenizer):
    # ori_model.cuda()
    model.eval()
    ori_model.eval()
    max_length = 2048   # fix model max length
    testenc = testenc.input_ids

    nsamples = testenc.numel() // max_length

    dev = next(model.parameters()).device

    testenc = testenc.to(dev)    
    prefill_acc = []
    prefill_mse = []
    decode_acc = []
    seq_len = 541   

    mse_fn = torch.nn.MSELoss(reduction='mean')
    hidden_mse = []
    # --------------prefill to first token---------------
    for i in tqdm(range(nsamples)):
        # 右侧是开区间
        batch = testenc[:, (i * max_length): (i * max_length + seq_len)]
        with torch.no_grad():
            model.cpu()
            ori_model.cuda()
            lm_logits = ori_model(batch).logits
            ori_model.cpu()
            model.cuda()
            qlm_logits = model(batch).logits   
        pred = lm_logits.argmax(dim=-1)        # (bsz, seq_len)
        q_pred = qlm_logits.argmax(dim=-1)
        loss = mse_fn(lm_logits, qlm_logits)
        prefill_mse.append(loss.item())
        prefill_acc.append((pred == q_pred).sum().item())
        loss = mse_fn(lm_logits[:,-1,:], qlm_logits[:,-1,:])
        hidden_mse.append(loss)

        del lm_logits, qlm_logits, loss
        torch.cuda.empty_cache()     
    prefill_acc = sum(prefill_acc) / (nsamples * seq_len)
    prefill_mse = sum(prefill_mse) / len(prefill_mse)
    hidden_mse = sum(hidden_mse) / len(hidden_mse)

    # ------------several tokens------------------
    new_tokens = 8
    pad_token_id = tokenizer.pad_token_id
    k_mse = []
    v_mse = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * max_length): (i * max_length + seq_len)]
        attn_mask = (batch != pad_token_id).long()
        
        model.cpu()
        ori_model.cuda()
        out = ori_model(batch, use_cache=True)
        ori_model.cpu()
        model.cuda()
        q_out = model(batch, use_cache=True)

        past_key_values = out.past_key_values
        q_past_key_values = q_out.past_key_values

        mse_k, mse_v = mse_between_pkv(past_key_values, q_past_key_values)
        k_mse.append(mse_k)
        v_mse.append(mse_v)

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
        q_next_token = torch.argmax(q_out.logits[:, -1, :], dim=-1)
        correct = 0 
        if next_token != q_next_token:
            continue 
        correct += 1
        ones = torch.ones(1, 1, dtype=attn_mask.dtype, device=dev)
        attn_mask = torch.cat([attn_mask, ones], dim=1) 
          
        next_token = next_token.unsqueeze(-1)
        q_next_token = q_next_token.unsqueeze(-1)        

        for i in range(new_tokens-1):                            
            past_seen_tokens = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
            pos_id = torch.arange(
                past_seen_tokens, past_seen_tokens + 1, device=dev
            ).unsqueeze(0)
            model.cpu()
            ori_model.cuda()
            out = ori_model(input_ids = next_token, attention_mask=attn_mask.clone(), position_ids=pos_id, use_cache=True, past_key_values=past_key_values)
            ori_model.cpu()
            model.cuda()
            q_out = model(input_ids = q_next_token, attention_mask=attn_mask.clone(), position_ids=pos_id, use_cache=True, past_key_values=q_past_key_values)

            loss = mse_fn(q_out.logits, out.logits)
            
            past_key_values = out.past_key_values
            q_past_key_values = q_out.past_key_values
            
            if next_token != q_next_token:
                break
            correct += 1
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
            q_next_token = torch.argmax(q_out.logits[:, -1, :], dim=-1)

            next_token = next_token.unsqueeze(-1)
            q_next_token = q_next_token.unsqueeze(-1)

            ones = torch.ones(1, 1, dtype=attn_mask.dtype, device=dev)
            attn_mask = torch.cat([attn_mask, ones], dim=1)

        decode_acc.append(correct / new_tokens)            

    decode_acc = sum(decode_acc) / (nsamples * new_tokens)
    k_mse = sum(k_mse) / len(k_mse)
    v_mse = sum(v_mse) / len(v_mse)

    ori_model.cpu()
    return prefill_acc, decode_acc, hidden_mse, k_mse, v_mse




def mse_eval_qk_internlm(model, testenc):
    max_length = 2048   # fix model max length
    testenc = testenc.input_ids

    nsamples = testenc.numel() // max_length

    dev = next(model.parameters()).device

    testenc = testenc.to(dev)    
    seq_len = 541   

    # --------------prefill to first token---------------
    for layer in model.model.layers:
        layer.attention.mse_cnt = 0
        layer.attention.k_mse = 0.0
        layer.attention.v_mse = 0.0
    
    
    for i in tqdm(range(nsamples)):
        # 右侧是开区间
        batch = testenc[:, (i * max_length): (i * max_length + seq_len)]    
        with torch.no_grad():
            out = model(batch).logits
    v_mse = []
    k_mse = []
    for layer in model.model.layers:
        assert layer.attention.mse_cnt == nsamples
        k_mse.append(layer.attention.k_mse / layer.attention.mse_cnt)        
        v_mse.append(layer.attention.v_mse / layer.attention.mse_cnt) 
    
    assert len(k_mse) == len(v_mse) == len(model.model.layers)       
    return sum(k_mse) / len(k_mse), sum(v_mse) / len(v_mse) 

def mse_eval_qk_qwen(model, testenc):
    max_length = 2048   # fix model max length
    testenc = testenc.input_ids

    nsamples = testenc.numel() // max_length

    dev = next(model.parameters()).device

    testenc = testenc.to(dev)    
    seq_len = 541   

    # --------------prefill to first token---------------
    for layer in model.model.layers:
        layer.self_attn.mse_cnt = 0
        layer.self_attn.k_mse = 0.0
        layer.self_attn.v_mse = 0.0
    
  
    for i in tqdm(range(nsamples)):
        # 右侧是开区间
        batch = testenc[:, (i * max_length): (i * max_length + seq_len)]    
        with torch.no_grad():
            out = model(batch).logits
    v_mse = []
    k_mse = []
    for layer in model.model.layers:
        assert layer.self_attn.mse_cnt == nsamples
        k_mse.append(layer.self_attn.k_mse / layer.self_attn.mse_cnt)        
        v_mse.append(layer.self_attn.v_mse / layer.self_attn.mse_cnt)        
    return sum(k_mse) / len(k_mse), sum(v_mse) / len(v_mse) 