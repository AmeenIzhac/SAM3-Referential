#!/usr/bin/env python3
"""Write a packed k-bit SAM3 checkpoint from the fp32 trainer checkpoint."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/reasonseg/quant/code")
import wquant                                     # noqa: E402
from sam3_quant_model import MMR_CKPT, _empty_model, load_fp_checkpoint, DTYPES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=MMR_CKPT)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--include-embeddings", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--dense-dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--device", default="cuda:0", help="device to run the fit on")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    model = load_fp_checkpoint(_empty_model(), args.checkpoint).to(args.device).eval()
    sd, cfg = wquant.build_quant_state_dict(
        model, args.bits, args.group_size, args.include_embeddings,
        tuple(args.skip), tuple(args.only), DTYPES[args.dense_dtype])
    cfg["source_checkpoint"] = args.checkpoint
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": sd, "config": cfg}, args.out)

    size = os.path.getsize(args.out)
    fp32 = sum(p.numel() for p in model.parameters()) * 4
    print(json.dumps({"out": args.out, "bits": args.bits, "group_size": args.group_size,
                      "n_quantized_tensors": len(cfg["quantized"]),
                      "coverage": round(cfg["quantized_params"] / cfg["total_params"], 4),
                      "bytes_quantized": cfg["bytes_quantized"],
                      "bytes_dense": cfg["bytes_dense"],
                      "file_bytes": size, "file_MB": round(size / 1e6, 1),
                      "vs_fp32_x": round(fp32 / size, 2),
                      "seconds": round(time.time() - t0, 1)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
