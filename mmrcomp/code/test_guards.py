#!/usr/bin/env python3
"""Prove the training guards actually fire.

Run 1 of the MMR ladder burned 15 h because the trainer silently stopped applying
optimizer steps. The guards added afterwards are only worth anything if they
trigger, so this exercises the real Trainer methods (bound to a stand-in `self`,
no GPU needed) rather than trusting that they look right.

    /workspace/envs/sam3/bin/python mmrcomp/code/test_guards.py
"""
import os
import sys
import types

sys.path.insert(0, "/workspace/newsam/sam3")
import torch
from sam3.train.trainer import Trainer


def stub():
    """Minimal stand-in for `self`: the periodic skip-rate log reads self.scaler."""
    return types.SimpleNamespace(
        scaler=types.SimpleNamespace(is_enabled=lambda: False, get_scale=lambda: 1.0)
    )

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


print("1. skip-rate abort")
os.environ["NEWSAM_SKIP_WINDOW"] = "50"
os.environ["NEWSAM_MAX_SKIP_RATE"] = "0.5"

# a healthy run must never trip it, however long it goes
ns = stub()
for i in range(500):
    Trainer._track_skip_rate(ns, False, i)
check("500 clean steps do not abort", True)

# every step skipped -> must abort, and must escape the (FloatingPointError,
# ValueError) handler that train_epoch wraps the step in
ns = stub()
raised = None
for i in range(500):
    try:
        Trainer._track_skip_rate(ns, True, i)
    except BaseException as e:  # noqa: BLE001
        raised = e
        break
check("100% skip rate aborts", raised is not None, f"at step {i}")
check(
    "abort escapes the per-batch handler",
    raised is not None and not isinstance(raised, (FloatingPointError, ValueError)),
    type(raised).__name__ if raised else "never raised",
)

# just under the threshold must NOT abort (no false positives)
ns = stub()
tripped = False
try:
    for i in range(400):
        Trainer._track_skip_rate(ns, i % 5 < 2, i)  # 40 % skipped
except BaseException:  # noqa: BLE001
    tripped = True
check("40% skip rate does not abort", not tripped)

print("2. weights fingerprint")
ns_m = stub(); ns = types.SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(8, 8),
                                                     torch.nn.Linear(8, 4)))
fp0 = Trainer._weights_fingerprint(ns)
fp1 = Trainer._weights_fingerprint(ns)
check("identical weights compare equal", int((fp0 != fp1).sum()) == 0)

with torch.no_grad():  # a single realistic-size update to one tensor
    ns.model[0].weight[0, 0] += 1e-4
fp2 = Trainer._weights_fingerprint(ns)
check("a 1e-4 change is detected", int((fp0 != fp2).sum()) > 0,
      f"{int((fp0 != fp2).sum())} tensor(s) moved")

# the same change under the ORIGINAL summed fingerprint, with SAM3's real ~1e19
# outlier tensor present -- this is why the summed version was replaced
big = torch.full((4,), 9.585e18)
summed_before = float(torch.stack([t.float().norm() for t in
                                   [big, ns.model[0].weight.detach(), ns.model[1].weight.detach()]]).sum())
with torch.no_grad():
    ns.model[0].weight[0, 1] += 1e-4
summed_after = float(torch.stack([t.float().norm() for t in
                                  [big, ns.model[0].weight.detach(), ns.model[1].weight.detach()]]).sum())
check("summed fingerprint would have MISSED it (why it was replaced)",
      summed_before == summed_after, "identical to the last bit at 1e19 scale")

print("\n" + ("ALL GUARDS FIRE" if ok else "SOME GUARDS DID NOT FIRE"))
sys.exit(0 if ok else 1)
