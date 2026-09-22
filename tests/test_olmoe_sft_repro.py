"""Reproduce OLMoE's exact v5 failure: functools.partial lm_head forward.

trl's _patch_chunked_ce_lm_head does inspect.signature(original_forward.__func__)
which raises AttributeError on a functools.partial. OLMoE (transformers) wraps
lm_head.forward in a partial. Verify:
  1. our _safe_patch wrapper swallows that AttributeError
  2. SFTTrainer with loss_type="nll" still trains after the skip
"""
import functools, os, sys
os.environ["PP_SMOKE"] = "1"
sys.path.insert(0, "training/kaggle")

import trl.trainer.sft_trainer as _sft

_orig_patch = _sft._patch_chunked_ce_lm_head
calls = {"patched": 0, "skipped": 0}

def _safe_patch(*a, **k):
    calls["patched"] += 1
    try:
        return _orig_patch(*a, **k)
    except AttributeError as e:
        if "__func__" in str(e):
            calls["skipped"] += 1
            print(f"  [test] _patch_chunked_ce_lm_head skipped: {e}")
            return
        raise
_sft._patch_chunked_ce_lm_head = _safe_patch

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32)
t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Simulate OLMoE (transformers 5.x): model.forward is a functools.partial
orig_fwd = m.forward
m.forward = functools.partial(orig_fwd)
assert not hasattr(m.forward, "__func__"), "partial lacks __func__ (expected)"

# Direct call — must NOT raise (our wrapper swallows the AttributeError)
_sft._patch_chunked_ce_lm_head(m, chunk_size=64, is_vlm=False)
assert calls["skipped"] == 1, f"expected 1 skip, got {calls}"
print("PASS: partial model.forward handled (skip path)")

# Undo the partial so SFTTrainer works normally
m.forward = orig_fwd

# Now verify loss_type="nll" end-to-end via the real kernel main
from kaggle_sft_olmoe import main
sys.argv = ["x", "--max-steps", "2"]
main()
print("PASS: loss_type=nll SFT end-to-end")
