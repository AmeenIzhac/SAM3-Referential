"""Weight-only group-wise RTN quantization for SAM3 checkpoints.

Scheme (the GPTQ/AWQ storage convention, round-to-nearest fit):
  * every nn.Linear weight (out, in) is split along the INPUT dim into groups of
    `group_size`; each group gets its own fp16 `scale` and uint8 integer `zero`
  * asymmetric min/max fit with 0 forced into the range, so w=0 is exactly
    representable:  q = clamp(round(w/scale) + zero, 0, 2**bits-1)
                    w_hat = (q - zero) * scale
  * `scale` is rounded to its stored dtype BEFORE the codes are computed, so the
    artifact on disk dequantizes bit-for-bit to the tensors that were evaluated
  * scales are stored in **bf16, not fp16**. Same 2 bytes, but bf16 carries
    fp32's exponent range. fp16 tops out at 65504, and the stock facebook/sam3
    checkpoint ships two ~-9.6e18 entries in the (unused, never-initialized)
    `text_projection` parameter -- enough to make that group's scale 3.8e16,
    which overflows fp16 to `inf` and then dequantizes as `0 * inf = NaN`,
    silently. Measured cost of bf16 over fp16 on real SAM3 tensors: +0.5 %
    relative error at 8 bits, +0.0 % at 4. See test_wquant.py.
  * codes are bit-packed at exactly `bits` bits/weight by a bit-plane transpose
    over groups of 8 values (8 values x k bits = k bytes), which works for any
    k in 1..8 -- including the 5 and 6 bit widths no stock kernel supports

Two runtime modes, numerically identical (asserted by verify_roundtrip):
  dense  -- unpack once at load into ordinary nn.Linear weights (full speed)
  packed -- keep the uint8 buffers resident and dequantize inside forward
            (low VRAM, slower; `weight` is a property so SAM3's fused
            addmm_act path, which reads linear.weight/.bias, still works)
"""
import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "sam3-wquant-v1"
# 2 bytes like fp16, but with fp32's exponent range -- see the module docstring
# for the SAM3 tensor that makes this the difference between a number and a NaN.
SCALE_DTYPE = torch.bfloat16


# ---------------------------------------------------------------- bit packing
def pack_bits(codes: torch.Tensor, bits: int):
    """codes: 1-D uint8, values in [0, 2**bits). -> (packed uint8 1-D, n_values).

    Bit-plane layout: values are taken 8 at a time and byte j of the output
    holds bit j of each of those 8 values. 8 values -> exactly `bits` bytes.
    """
    assert codes.dtype == torch.uint8 and codes.dim() == 1
    n = codes.numel()
    pad = (-n) % 8
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad)])
    c = codes.view(-1, 8).to(torch.int32)
    idx = torch.arange(8, device=c.device, dtype=torch.int32)
    out = torch.empty(c.shape[0], bits, dtype=torch.uint8, device=c.device)
    for j in range(bits):
        out[:, j] = (((c >> j) & 1) << idx).sum(dim=1).to(torch.uint8)
    return out.reshape(-1).contiguous(), n


def unpack_bits(packed: torch.Tensor, n: int, bits: int) -> torch.Tensor:
    """Inverse of pack_bits. -> 1-D uint8 of length n."""
    b = packed.view(-1, bits).to(torch.int32)
    idx = torch.arange(8, device=b.device, dtype=torch.int32)
    v = torch.zeros(b.shape[0], 8, dtype=torch.int32, device=b.device)
    for j in range(bits):
        v |= ((b[:, j].unsqueeze(1) >> idx) & 1) << j
    return v.reshape(-1)[:n].to(torch.uint8)


# ------------------------------------------------------------- quantize / not
def _gsize(group_size, in_features):
    """group_size <= 0 means per-output-channel: one scale/zero for the whole
    row, i.e. the granularity every stock int8 path uses."""
    return in_features if group_size <= 0 else min(group_size, in_features)


def quantize_tensor(W: torch.Tensor, bits: int, group_size: int):
    """W: (out, in) float. -> (codes uint8 (out, ng*g), scale fp16, zero uint8, n_pad).

    Groups run along the input dim. A ragged final group is handled by padding
    with a replicate of the last column, which cannot change that group's
    min/max, and the padding is dropped again on dequantize.
    """
    assert W.dim() == 2
    out_f, in_f = W.shape
    g = _gsize(group_size, in_f)
    ng = (in_f + g - 1) // g
    pad = ng * g - in_f
    Wp = W if pad == 0 else torch.cat([W, W[:, -1:].expand(out_f, pad)], dim=1)
    Wg = Wp.reshape(out_f, ng, g).float()

    qmax = (1 << bits) - 1
    zero_t = torch.zeros(1, device=W.device)
    wmin = torch.minimum(Wg.amin(dim=2), zero_t)
    wmax = torch.maximum(Wg.amax(dim=2), zero_t)
    scale = (wmax - wmin) / qmax
    dead = scale <= 0                       # all-zero group
    scale = torch.where(dead, torch.ones_like(scale), scale)
    # round the scale to its stored precision first, so the codes are fit against
    # the value dequantize will actually use
    scale = scale.to(SCALE_DTYPE).float()
    if not torch.isfinite(scale).all():
        # Reachable two ways: a group whose (max - min) overflows fp32 outright,
        # or -- if SCALE_DTYPE is narrowed back to fp16 -- any group whose range
        # exceeds 65504*qmax. Either way, refuse rather than return a NaN.
        raise ValueError(
            f"non-finite group scale for a {tuple(W.shape)} tensor "
            f"(max|w|={W.abs().max().item():.3e}, scale dtype {SCALE_DTYPE}): "
            f"its dynamic range cannot be represented")
    zero = torch.clamp(torch.round(-wmin / scale), 0, qmax)
    zero = torch.where(dead, torch.zeros_like(zero), zero)
    codes = torch.clamp(torch.round(Wg / scale.unsqueeze(2)) + zero.unsqueeze(2), 0, qmax)
    return (codes.to(torch.uint8).reshape(out_f, ng * g),
            scale.to(SCALE_DTYPE), zero.to(torch.uint8), pad)


def dequantize_tensor(codes, scale, zero, in_f, group_size, dtype=torch.float32):
    """codes uint8 (out, ng*g) -> W (out, in_f) in `dtype`."""
    out_f = codes.shape[0]
    g = _gsize(group_size, in_f)
    ng = codes.shape[1] // g
    q = codes.view(out_f, ng, g).to(torch.float32)
    w = (q - zero.float().unsqueeze(2)) * scale.float().unsqueeze(2)
    return w.reshape(out_f, ng * g)[:, :in_f].to(dtype)


# ------------------------------------------------------------------ selection
# Selection works on *parameters*, not module types, on purpose. SAM3's CLIP text
# encoder is built from nn.MultiheadAttention, whose in_proj_weight is a raw
# Parameter -- a scan for nn.Linear misses 24 x 3.1M = 75M params of language
# attention, the part of the model that actually reads the reasoning question.
EMBED_SUFFIXES = ("token_embedding.weight",)


def is_embedding(name):
    return name.endswith(EMBED_SUFFIXES)


def select_params(model, group_size, include_embeddings=False, skip=(), only=(),
                  min_numel=1 << 16):
    """Every 2-D float parameter worth quantizing, as (name, param).

    Skipped: tensors narrower than one group (a handful of 2/4-input geometry
    projections -- group-wise quantization is meaningless there), and anything
    under `min_numel` (65K, i.e. every LayerNorm/bias/small head), where the
    per-group scale+zero overhead is a large fraction of the tensor and the
    accuracy cost buys no compression.
    """
    skip_res = [re.compile(p) for p in skip]
    only_res = [re.compile(p) for p in only]
    out = []
    for name, p in model.named_parameters():
        if p.dim() != 2 or not p.is_floating_point():
            continue
        if p.numel() < min_numel or (group_size > 0 and p.shape[1] < group_size):
            continue
        if is_embedding(name) and not include_embeddings:
            continue
        if any(r.search(name) for r in skip_res):
            continue
        if only_res and not any(r.search(name) for r in only_res):
            continue
        out.append((name, p))
    return out


# --------------------------------------------------------------- quant module
class QuantLinear(nn.Module):
    """nn.Linear with packed k-bit weights, dequantized inside forward.

    `weight` is a property rather than a Parameter so call sites that read
    `linear.weight` (SAM3's fused addmm_act in the ViT MLP) keep working.
    Used only by mode='packed'; the evaluated `dense` mode needs no custom
    module, since it produces ordinary weights that are bit-identical to what
    this would compute.
    """

    def __init__(self, in_features, out_features, bits, group_size, bias=True,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = bits, group_size
        g = _gsize(group_size, in_features)
        ng = (in_features + g - 1) // g
        self.n_codes = out_features * ng * g
        nbytes = ((self.n_codes + 7) // 8) * bits
        self.register_buffer("qweight", torch.zeros(nbytes, dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.zeros(out_features, ng, dtype=SCALE_DTYPE, device=device))
        self.register_buffer("qzeros", torch.zeros(out_features, ng, dtype=torch.uint8, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None
        self._dtype = dtype

    @property
    def weight(self):
        codes = unpack_bits(self.qweight, self.n_codes, self.bits).view(self.out_features, -1)
        return dequantize_tensor(codes, self.scales, self.qzeros,
                                 self.in_features, self.group_size, self._dtype)

    def forward(self, x):
        w = self.weight
        return F.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.bits}, group={self.group_size}")


def _set_module(model, name, new):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


# ------------------------------------------------------------------ top level
def quantize_model_inplace(model, bits, group_size, include_embeddings=False,
                           skip=(), only=(), verbose=True):
    """Round-trip every selected weight through the quantizer, writing the
    dequantized values back in place (`dense` mode). Returns a stats dict."""
    params = select_params(model, group_size, include_embeddings, skip, only)
    n_q = 0
    errs = []
    with torch.no_grad():
        for name, p in params:
            W = p.data
            codes, scale, zero, _ = quantize_tensor(W.float(), bits, group_size)
            Wq = dequantize_tensor(codes, scale, zero, W.shape[1], group_size, W.dtype)
            # A silent NaN here is the failure mode that cost us an afternoon:
            # it does not raise, does not change the loss, and only shows up as
            # a garbage metric. Refuse to hand back a tensor that has one.
            if not torch.isfinite(Wq).all():
                raise ValueError(f"{name}: quantization produced "
                                 f"{int((~torch.isfinite(Wq)).sum())} non-finite values")
            rel = (Wq.float() - W.float()).norm() / W.float().norm().clamp_min(1e-12)
            errs.append((name, float(rel)))
            p.data.copy_(Wq)
            n_q += W.numel()
    tot = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[wquant] {bits}-bit g{group_size}: quantized {len(params)} tensors, "
              f"{n_q/1e6:.1f}M / {tot/1e6:.1f}M params ({100*n_q/tot:.1f}%)", flush=True)
    return {"bits": bits, "group_size": group_size, "n_tensors": len(params),
            "skip": list(skip), "only": list(only),
            "quantized_params": n_q, "total_params": tot,
            "coverage": round(n_q / tot, 4),
            "rel_err_mean": float(sum(e for _, e in errs) / max(1, len(errs))),
            "rel_err_median": float(sorted(e for _, e in errs)[len(errs) // 2]) if errs else 0.0,
            "rel_err_max": max((e for _, e in errs), default=0.0),
            "rel_err_max_tensor": max(errs, key=lambda t: t[1])[0] if errs else None,
            "per_tensor_rel_err": dict(errs)}


def build_quant_state_dict(model, bits, group_size, include_embeddings=False,
                           skip=(), only=(), dense_dtype=torch.float32):
    """Serializable packed artifact: packed codes for the selected tensors,
    everything else cast to `dense_dtype` (complex/int tensors left alone).

    dense_dtype defaults to fp32, NOT bf16, so that loading the artifact
    reproduces the evaluated model bit-for-bit -- the ~15M unquantized params
    (LayerNorms, biases, conv kernels, pos_embed) are what the eval ran on, and
    silently rounding them in the artifact would make the measured accuracy
    belong to a model nobody shipped. Pass bfloat16 to trade that for ~30 MB.
    """
    params = dict(select_params(model, group_size, include_embeddings, skip, only))
    sd = {}
    cfg = {"format": FORMAT, "bits": bits, "group_size": group_size,
           "dense_dtype": str(dense_dtype).replace("torch.", ""),
           "include_embeddings": include_embeddings, "skip": list(skip),
           "only": list(only),
           "quantized": {}}
    q_bytes = d_bytes = 0
    for name, p in params.items():
        W = p.data.float()
        codes, scale, zero, _ = quantize_tensor(W, bits, group_size)
        chk = dequantize_tensor(codes, scale, zero, W.shape[1], group_size)
        if not torch.isfinite(chk).all():
            raise ValueError(f"{name}: quantization produced non-finite values")
        packed, n = pack_bits(codes.reshape(-1).contiguous(), bits)
        sd[f"{name}::q"] = packed.cpu()
        sd[f"{name}::s"] = scale.cpu()
        sd[f"{name}::z"] = zero.cpu()
        cfg["quantized"][name] = {"shape": list(W.shape), "n_codes": n}
        q_bytes += packed.numel() + scale.numel() * 2 + zero.numel()
    for k, v in model.state_dict().items():
        if k in params:
            continue
        vv = v.to(dense_dtype) if v.is_floating_point() else v
        sd[k] = vv.cpu()
        d_bytes += vv.numel() * vv.element_size()
    cfg["bytes_quantized"] = q_bytes
    cfg["bytes_dense"] = d_bytes
    cfg["quantized_params"] = sum(p.numel() for p in params.values())
    cfg["total_params"] = sum(p.numel() for p in model.parameters())
    return sd, cfg


def load_quantized(model, sd, cfg, mode="dense", device="cpu"):
    """Load a packed artifact into `model`.

    mode='dense'  -- dequantize into ordinary parameters (full-speed inference;
                     numerically identical to quantize_model_inplace)
    mode='packed' -- install QuantLinear for every quantized nn.Linear and keep
                     its uint8 codes resident; any quantized tensor that is not
                     a Linear weight (MHA in_proj, embeddings) falls back to
                     dense, since it has no module to wrap.
    """
    assert cfg["format"] == FORMAT, cfg.get("format")
    bits, g = cfg["bits"], cfg["group_size"]

    dense = {k: v for k, v in sd.items() if "::" not in k}
    # keys written straight into a replaced module rather than via load_state_dict;
    # they must not be reported missing below
    handled = set()
    n_packed = 0
    for name, info in cfg["quantized"].items():
        out_f, in_f = info["shape"]
        codes = unpack_bits(sd[f"{name}::q"].to(device), info["n_codes"], bits).view(out_f, -1)
        scale, zero = sd[f"{name}::s"].to(device), sd[f"{name}::z"].to(device)
        mod_name = name.rsplit(".", 1)[0]
        mod = model.get_submodule(mod_name) if name.endswith(".weight") else None
        if mode == "packed" and isinstance(mod, nn.Linear):
            ql = QuantLinear(in_f, out_f, bits, g, bias=mod.bias is not None)
            ql.qweight.data = sd[f"{name}::q"].cpu()
            ql.scales.data = scale.cpu()
            ql.qzeros.data = zero.cpu()
            if ql.bias is not None:
                ql.bias.data = dense.pop(f"{mod_name}.bias").float().cpu()
                handled.add(f"{mod_name}.bias")
            _set_module(model, mod_name, ql)
            n_packed += 1
        else:
            dense[name] = dequantize_tensor(codes, scale, zero, in_f, g,
                                            torch.float32).cpu()

    tgt = model.state_dict()
    fixed = {}
    for k, v in dense.items():
        ref = tgt.get(k)
        fixed[k] = (v.to(ref.dtype) if (ref is not None and ref.is_floating_point()
                                        and v.is_floating_point()) else v)
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    missing = [m for m in missing
               if m not in handled and not m.endswith((".qweight", ".scales", ".qzeros"))]
    if missing or unexpected:
        raise RuntimeError(f"quant load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model, n_packed


def verify_roundtrip(W, bits, group_size):
    """quantize -> pack -> unpack -> dequantize must be bit-identical to
    quantize -> dequantize."""
    codes, scale, zero, _ = quantize_tensor(W.float(), bits, group_size)
    a = dequantize_tensor(codes, scale, zero, W.shape[1], group_size)
    packed, n = pack_bits(codes.reshape(-1).contiguous(), bits)
    codes2 = unpack_bits(packed, n, bits).view(codes.shape)
    b = dequantize_tensor(codes2, scale, zero, W.shape[1], group_size)
    return bool(torch.equal(codes, codes2)) and bool(torch.equal(a, b))
