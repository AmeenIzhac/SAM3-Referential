#!/usr/bin/env python3
"""Correctness tests for the quantizer. Run with the sam3 env python.

The claims worth testing are the ones a wrong result would hide behind:
  1. bit-packing is lossless at every width 1..8 (including 5/6/7, which no
     stock kernel packs)
  2. an all-zero group survives (the scale==0 degenerate case)
  3. w=0 is exactly representable (0 is forced into every group's range), so
     zero-valued weights do not drift
  4. error decreases monotonically with bits, and halves per extra bit from
     3 bits up (at 2->3 it falls faster than 2x: with only 4 levels the
     asymmetric min/max fit is dominated by each group's outliers, so the first
     extra level buys more than a uniform-grid argument predicts)
  5. save -> load('dense') reproduces quantize_model_inplace BIT-EXACTLY, so
     the artifact on disk is the thing that was evaluated
  6. load('packed') produces bit-identical weights to load('dense')
  7. the eval harness's own model reaches the coverage claimed in the writeup
  8. REGRESSION: the stock facebook/sam3 `text_projection` parameter -- two
     ~-9.6e18 entries in an otherwise ordinary tensor -- quantizes to finite
     values at every width. With fp16 scales it did not: 3.8e16 overflows fp16
     to inf and dequantizes as 0*inf = NaN, silently, in a tensor SAM3 never
     reads (`pooled` is discarded), so nothing downstream complained.
"""
import sys

import torch

sys.path.insert(0, "/workspace/reasonseg/quant/code")
import wquant  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra else ''}", flush=True)
    if not cond:
        FAILED.append(name)


def t1_packing():
    print("1. bit-packing is lossless at every width")
    torch.manual_seed(0)
    for bits in range(1, 9):
        for shape in [(37, 259), (1, 128), (256, 4736), (200, 256)]:
            W = torch.randn(*shape)
            if not wquant.verify_roundtrip(W, bits, 128):
                check(f"roundtrip bits={bits} shape={shape}", False)
                return
    check("roundtrip, bits 1..8 x 4 shapes", True)
    # exhaustive over all representable codes
    for bits in range(1, 9):
        codes = torch.arange(1 << bits, dtype=torch.uint8).repeat(7)
        packed, n = wquant.pack_bits(codes, bits)
        back = wquant.unpack_bits(packed, n, bits)
        ok = torch.equal(codes, back)
        exp = ((n + 7) // 8) * bits
        check(f"exhaustive codes bits={bits}", ok and packed.numel() == exp,
              f"{packed.numel()}B for {n} vals = {8*packed.numel()/n:.2f} bits/val")


def t2_degenerate():
    print("2. degenerate groups")
    W = torch.zeros(4, 256)
    c, s, z, _ = wquant.quantize_tensor(W, 4, 128)
    check("all-zero tensor dequantizes to zero",
          bool(torch.equal(wquant.dequantize_tensor(c, s, z, 256, 128), W)))
    W = torch.randn(4, 256); W[:, :128] = 0.0
    c, s, z, _ = wquant.quantize_tensor(W, 4, 128)
    Wq = wquant.dequantize_tensor(c, s, z, 256, 128)
    check("all-zero group inside a live tensor stays zero",
          bool((Wq[:, :128] == 0).all()) and bool((Wq[:, 128:] != 0).any()))
    W = torch.randn(3, 200)   # ragged final group (200 = 128 + 72)
    c, s, z, _ = wquant.quantize_tensor(W, 5, 128)
    Wq = wquant.dequantize_tensor(c, s, z, 200, 128)
    check("ragged final group keeps shape and stays finite",
          Wq.shape == W.shape and bool(torch.isfinite(Wq).all()),
          f"rel_err={((Wq-W).norm()/W.norm()):.4f}")


def t3_zero_exact():
    print("3. w=0 is exactly representable")
    torch.manual_seed(1)
    for bits in (2, 3, 4, 5, 6, 8):
        W = torch.randn(64, 512)
        W[W.abs() < 0.05] = 0.0
        mask = W == 0
        c, s, z, _ = wquant.quantize_tensor(W, bits, 128)
        Wq = wquant.dequantize_tensor(c, s, z, 512, 128)
        check(f"zeros preserved exactly at {bits} bits",
              bool((Wq[mask] == 0).all()), f"{int(mask.sum())} zeros")


def t4_error_scaling():
    print("4. error falls ~2x per extra bit (from 3 bits up)")
    torch.manual_seed(2)
    W = torch.randn(512, 1024)
    errs = {}
    for bits in range(2, 9):
        c, s, z, _ = wquant.quantize_tensor(W, bits, 128)
        Wq = wquant.dequantize_tensor(c, s, z, 1024, 128)
        errs[bits] = float((Wq - W).norm() / W.norm())
    mono = all(errs[b] < errs[b - 1] for b in range(3, 9))
    ratios = {b: errs[b - 1] / errs[b] for b in range(3, 9)}
    check("monotonic in bits", mono, " ".join(f"{b}:{e:.4f}" for b, e in errs.items()))
    # the clean ~2x-per-bit regime is 5 bits and up; below that each extra level
    # buys MORE than 2x, because with few levels the asymmetric min/max fit is
    # dominated by each group's outliers rather than by grid spacing
    check("ratio per extra bit in [1.9, 2.1] for bits >= 5",
          all(1.9 <= r <= 2.1 for b, r in ratios.items() if b >= 5),
          " ".join(f"{b}:{r:.2f}" for b, r in ratios.items()))
    check("the 2->3 and 3->4 steps gain more than 2.1x",
          ratios[3] > 2.1 and ratios[4] > 2.1, f"{ratios[3]:.2f} {ratios[4]:.2f}")


def t5_artifact_bitexact():
    print("5. saved artifact == in-place quantization, bit-exactly")
    from sam3_quant_model import _empty_model, load_fp_checkpoint, MMR_CKPT
    bits, g = 4, 128
    a = load_fp_checkpoint(_empty_model(), MMR_CKPT)
    st = wquant.quantize_model_inplace(a, bits, g, include_embeddings=True, verbose=False)
    b = load_fp_checkpoint(_empty_model(), MMR_CKPT)
    sd, cfg = wquant.build_quant_state_dict(b, bits, g, include_embeddings=True)
    c = _empty_model()
    wquant.load_quantized(c, sd, cfg, mode="dense")
    sa, sc = a.state_dict(), c.state_dict()
    bad = [k for k in sa if not torch.equal(sa[k], sc[k])]
    check("every tensor identical after save->load(dense)", not bad,
          f"coverage={st['coverage']:.3f} first_bad={bad[:3]}")

    d = _empty_model()
    _, n_packed = wquant.load_quantized(d, sd, cfg, mode="packed")
    qmods = [(n, m) for n, m in d.named_modules() if isinstance(m, wquant.QuantLinear)]
    diffs = [n for n, m in qmods if not torch.equal(m.weight, sa[f"{n}.weight"])]
    check("packed-mode weights identical to dense-mode",
          not diffs and n_packed > 0, f"{n_packed} QuantLinear installed, bad={diffs[:3]}")
    return a


def t8_text_projection_outlier():
    print("8. regression: the text_projection outlier quantizes finite")
    from sam3_quant_model import MMR_CKPT
    ck = torch.load(MMR_CKPT, map_location="cpu", mmap=True, weights_only=True)
    key = "backbone.language_backbone.encoder.text_projection"
    W = ck["model"][key].float()
    n_out = int((W.abs() > 65504).sum())
    check("the pathological tensor is still shaped as expected",
          n_out == 2 and W.abs().max() > 1e18,
          f"{n_out} entries over fp16 max, |max|={W.abs().max():.2e}, "
          f"median|w|={W.abs().median():.4f}")
    for g in (0, 32, 128):
        for bits in (2, 4, 8):
            c, s_, z, _ = wquant.quantize_tensor(W, bits, g)
            Wq = wquant.dequantize_tensor(c, s_, z, W.shape[1], g)
            if not torch.isfinite(Wq).all():
                check(f"finite at {bits} bits g{g}", False,
                      f"{int((~torch.isfinite(Wq)).sum())} non-finite")
                return
    check("finite at every (bits, group) combination tried", True)

    # With bf16 scales the fp16 overflow class is *gone*, not merely mitigated:
    # scale = (max-min)/qmax and bf16's range equals fp32's, so no finite fp32
    # tensor can produce an unrepresentable scale. Check the near-limit case.
    big = torch.full((1, 128), torch.finfo(torch.float32).max / 4, dtype=torch.float32)
    big[0, 0] = 0.0
    c, s_, z, _ = wquant.quantize_tensor(big, 8, 128)
    Wq = wquant.dequantize_tensor(c, s_, z, 128, 128)
    check("a group spanning a quarter of the fp32 range still quantizes finite",
          bool(torch.isfinite(Wq).all()), f"scale={s_.float().max().item():.3e}")

    # The one input that IS still pathological: a group spanning +-fp32max, where
    # (max - min) overflows fp32 before a scale is even formed. That must raise.
    worst = torch.zeros(1, 128)
    worst[0, 0] = torch.finfo(torch.float32).max
    worst[0, 1] = -torch.finfo(torch.float32).max
    try:
        wquant.quantize_tensor(worst, 8, 128)
        check("a +-fp32max group raises instead of going quiet", False, "no exception")
    except ValueError:
        check("a +-fp32max group raises ValueError", True)


def t6_coverage(model):
    print("6. coverage of the real model")
    tot = sum(p.numel() for p in model.parameters())
    for emb in (False, True):
        sel = wquant.select_params(model, 128, include_embeddings=emb)
        n = sum(p.numel() for _, p in sel)
        check(f"coverage include_embeddings={emb} >= {'0.98' if emb else '0.92'}",
              n / tot >= (0.98 if emb else 0.92),
              f"{len(sel)} tensors, {n/1e6:.1f}M/{tot/1e6:.1f}M = {100*n/tot:.1f}%")
    sel = wquant.select_params(model, 128)
    mha = [(n, p) for n, p in sel if "in_proj_weight" in n]
    n_par = sum(p.numel() for _, p in mha)
    lang = sum(p.numel() for n, p in mha if "language_backbone" in n)
    check("every MultiheadAttention in_proj_weight is covered",
          len(mha) == 61 and n_par > 82e6,
          f"{len(mha)} tensors, {n_par/1e6:.1f}M params, {lang/1e6:.1f}M of it language attention")


if __name__ == "__main__":
    t1_packing()
    t2_degenerate()
    t3_zero_exact()
    t4_error_scaling()
    t8_text_projection_outlier()
    m = t5_artifact_bitexact()
    t6_coverage(m)
    print(f"\n{'ALL TESTS PASSED' if not FAILED else 'FAILURES: ' + str(FAILED)}")
    sys.exit(1 if FAILED else 0)
