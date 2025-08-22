import random
import numpy as np
import torch
import transformers

import logging

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def infer_accumulation_dtype(input_dtype: torch.dtype,
                             device: torch.device,
                             op: str = "matmul") -> torch.dtype:
    """
    Best-effort inference of accumulation dtype for common Torch ops.
    Returns a torch.dtype that the reduction/accumulation most likely uses.
    """
    if device.type == "cuda":
        if input_dtype in (torch.float16, torch.bfloat16):
            # Default: Tensor Cores accumulate in FP32
            # unless reduced-precision reduction is explicitly permitted.
            if input_dtype is torch.float16 and torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
                return torch.float16
            if input_dtype is torch.bfloat16 and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
                return torch.bfloat16
            return torch.float32

        if input_dtype is torch.float32:
            # TF32 affects *compute format* for F32 matmul/conv, not accumulator width.
            # Accumulation remains FP32.
            return torch.float32

        if input_dtype in (torch.int8, torch.qint8):
            # Quantized CUDA/kernels: int32 accumulation
            return torch.int32

    # CPU fallbacks / typical BLAS behavior
    if device.type == "cpu":
        if input_dtype in (torch.float16, torch.bfloat16, torch.float32):
            return torch.float32
        if input_dtype in (torch.int8, torch.qint8):
            return torch.int32

    # Sensible default if something exotic
    return torch.float32

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def skip_initialization():
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

def cleanup_memory(verbose=True) -> None:
    """Clear GPU memory by running garbage collection and emptying cache."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbose:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2/3/Qwen-2."""
    no_split_module_classes = ['LlamaDecoderLayer', 'Qwen2DecoderLayer']
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split_module_classes)

    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)

    dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=model.state_dict())
    cleanup_memory()


def seed_everything(seed=0) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    transformers.set_seed(seed)

