"""Build a SAM3 image model from a trainer checkpoint, optionally weight-quantized.

Self-contained builder (the old /workspace/newsam/benchmark_refexp is gone); this
is the same loader gold_eval.py uses, against the stock /workspace/sam3 tree.
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/sam3")
sys.path.insert(0, "/workspace/reasonseg/quant/code")

import wquant  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

MMR_CKPT = "/mnt/data0/ameen/mmr_runs/mmr_scale/checkpoints/checkpoint.pt"
BASE_CKPT = ("/workspace/.cache/huggingface/models--facebook--sam3/snapshots/"
             "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# The detection heads -- everything downstream of the two backbones. 27.5M params,
# 3.3 % of what gets quantized, but 21x the per-parameter precision sensitivity of
# the vision backbone (quant.md §6), which is what makes a mixed-precision pass
# worth having: hold these at 8 bits and the rest of the model can go lower.
HEAD_PATTERNS = ("^transformer", "^geometry_encoder", "^segmentation_head",
                 "^dot_prod_scoring")


def _empty_model():
    return build_sam3_image_model(checkpoint_path=None, load_from_HF=False,
                                  device="cpu", enable_inst_interactivity=False)


def load_fp_checkpoint(model, checkpoint):
    """Load either checkpoint shape this project produces or consumes.

    * **release** (`facebook/sam3`) — a full *video* checkpoint: image-model
      weights under `detector.`, video-tracker weights under `tracker.`. The
      tracker half has no home in an image model built with
      `enable_inst_interactivity=False`, so it is dropped, exactly as
      `model_builder._load_checkpoint` does.
    * **trainer** (this repo's fine-tunes) — a plain image-model state dict with
      no prefix at all, which is why `model_builder`'s loader cannot read them
      (its `if "detector" in k` filter would match nothing).

    Either way the load must be exact for the keys that remain: an unexpected or
    missing tensor is an error, never a silent skip.
    """
    ck = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    n_all = len(sd)
    release = any(k.startswith("detector.") for k in sd)
    if release:
        sd = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
        own = set(model.state_dict())
        extra = sorted(set(sd) - own)
        sd = {k: v for k, v in sd.items() if k in own}
        kind = (f"release: {n_all - len(sd) - len(extra)} tracker/video + {len(extra)} "
                f"interactive-branch tensors dropped")
    else:
        extra, kind = [], "trainer"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # `missing` is the property that matters: every parameter of the model we are
    # about to run must have been given a value. Extra tensors are only tolerated
    # for the release checkpoint, which carries whole sub-models (video tracker,
    # SAM2 interactive branch) that an enable_inst_interactivity=False image model
    # has no slot for -- and they are counted and reported, not silently ignored
    # the way model_builder._load_checkpoint does.
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch ({kind}): {len(missing)=} {len(unexpected)=}\n"
                         f"{missing[:5]=}\n{unexpected[:5]=}")
    print(f"[load] {kind}; {len(sd)} tensors into the image model", flush=True)
    return model


def cast_weights(model, dtype):
    """Cast every floating-point parameter/buffer to `dtype` and back to fp32
    storage, i.e. simulate shipping the model at that precision while keeping
    the fp32 compute container the baseline uses."""
    if dtype == torch.float32:
        return model
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.data.copy_(p.data.to(dtype).float())
        for b in model.buffers():
            if b.is_floating_point():
                b.data.copy_(b.data.to(dtype).float())
    return model


def build(checkpoint=MMR_CKPT, device="cuda:0", bits=None, group_size=128,
          include_embeddings=False, skip=(), weight_dtype="fp32",
          quant_ckpt=None, quant_mode="dense", only=(), head_bits=None,
          verbose=True):
    """Returns (model, info). Exactly one of `bits` / `quant_ckpt` may be set.

    head_bits: run a second in-place pass over HEAD_PATTERNS at that width while
    the main pass skips them -- i.e. mixed precision, heads held higher.
    """
    t0 = time.time()
    model = _empty_model()
    info = {"checkpoint": checkpoint, "weight_dtype": weight_dtype}

    if quant_ckpt:
        blob = torch.load(quant_ckpt, map_location="cpu", weights_only=False)
        cfg = blob["config"] if "config" in blob else json.loads(blob["config_json"])
        _, n_packed = wquant.load_quantized(model, blob["state_dict"], cfg, mode=quant_mode)
        info.update({"quant_ckpt": quant_ckpt, "quant_mode": quant_mode,
                     "bits": cfg["bits"], "group_size": cfg["group_size"],
                     "include_embeddings": cfg["include_embeddings"],
                     "n_quant_tensors": len(cfg["quantized"]),
                     "n_packed_modules": n_packed,
                     "coverage": round(cfg["quantized_params"] / cfg["total_params"], 4)})
    else:
        load_fp_checkpoint(model, checkpoint)
        if weight_dtype != "fp32":
            cast_weights(model, DTYPES[weight_dtype])
        if bits:
            body_skip = tuple(skip) + (HEAD_PATTERNS if head_bits else ())
            st = wquant.quantize_model_inplace(model, bits, group_size,
                                               include_embeddings, body_skip, only,
                                               verbose)
            st.pop("per_tensor_rel_err", None)
            info.update({k: v for k, v in st.items()})
            info["quant_mode"] = "dense"
            if head_bits:
                hst = wquant.quantize_model_inplace(model, head_bits, group_size,
                                                    False, (), HEAD_PATTERNS, verbose)
                hst.pop("per_tensor_rel_err", None)
                info["head_bits"] = head_bits
                info["head_n_tensors"] = hst["n_tensors"]
                info["head_quantized_params"] = hst["quantized_params"]
                info["quantized_params"] += hst["quantized_params"]
                info["coverage"] = round(info["quantized_params"] / hst["total_params"], 4)
                info["n_tensors"] += hst["n_tensors"]

    model = model.to(device).eval()
    info["build_seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"[build] {info}", flush=True)
    return model, info
