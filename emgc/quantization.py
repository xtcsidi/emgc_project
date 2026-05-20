import logging
import torch
import torch.nn as nn
from .datatypes import Precision

try:
    import bitsandbytes as bnb  # noqa: F401
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
    logging.warning("bitsandbytes not found – INT8/INT4 will use torch fake-quant fallback.")

log = logging.getLogger("MemoryGatedOptimizer")

def _get_parent(model: nn.Module, dotted_name: str):
    parts  = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def to_fp16(model: nn.Module) -> Precision:
    model.half()
    log.info("◀ Precision restored → FP16")
    return Precision.FP16

def to_int8(model: nn.Module) -> Precision:
    if _BNB_AVAILABLE:
        import bitsandbytes as bnb
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent, attr = _get_parent(model, name)
                q = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                )
                q.weight = module.weight
                if module.bias is not None:
                    q.bias = module.bias
                setattr(parent, attr, q)
    else:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                with torch.no_grad():
                    w   = m.weight.float()
                    scale = w.abs().max() / 127.0
                    m.weight.data = (w / scale).round().clamp(-128, 127) * scale
        model.half()

    log.info("▼ Precision → INT8")
    return Precision.INT8

def to_int4(model: nn.Module) -> Precision:
    if _BNB_AVAILABLE:
        from bitsandbytes.nn import Linear4bit
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent, attr = _get_parent(model, name)
                q = Linear4bit(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                    compress_statistics=True,
                )
                q.weight = module.weight
                if module.bias is not None:
                    q.bias = module.bias
                setattr(parent, attr, q)
    else:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                with torch.no_grad():
                    w     = m.weight.float()
                    scale = w.abs().max() / 7.0
                    m.weight.data = (w / scale).round().clamp(-8, 7) * scale
        model.half()

    log.info("▼▼ Precision → INT4")
    return Precision.INT4
