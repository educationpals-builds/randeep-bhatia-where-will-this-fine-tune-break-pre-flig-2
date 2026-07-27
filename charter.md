# Pre-Flight Charter: Fine-Tune Break Analysis

## Specimen Under Review

**Model:** 34-layer open decoder model, block code lifted from a tutorial repo, fine-tuning on 180k clinical intake notes, launch in 9 days.

**Run Reality:** 8×A100 for ~11 days, fp16 by default, 34 layers, sequence length 4096, nobody on the team has traced the forward pass end to end, and the repo's README says 'works out of the box'.

**Stakes:** A run that diverges at layer 30 burns the quarter's GPU budget and pushes the delivery date past the contract review.

---

## Standard Line

Safe to train means:
1. Every one of the six operations in the block is accounted for in the code we will actually run
2. Normalization sits before each sub-layer
3. The input path is preserved by addition in both halves
4. A 200-step smoke run shows gradients alive in the deepest layers

---

## Five Clause Findings

### Clause 1: Clean Copy of the Input

**Status:** ✓ CLEAR

**Evidence:** Line 142: `residual = x.clone()` before attn — clean copy preserved.

**Key:** The input tensor is explicitly cloned before any transformation. This prevents in-place mutation from corrupting the residual stream.

---

### Clause 2: Normalization Placement

**Status:** ✓ CLEAR

**Evidence:** LayerNorm runs before attn at line 118 (pre-norm).

**Key:** Pre-norm architecture confirmed. LayerNorm stabilizes activations before they enter attention, which is the modern standard for deep transformers.

---

### Clause 3: Where the Work Happens

**Status:** ✓ CLEAR

**Evidence:** FFN at lines 155-168 does 4x expand — work happens in MLP not attention.

**Key:** The feed-forward network performs the heavy lifting with a 4x expansion factor. Attention routes; FFN computes. This is correctly implemented.

---

### Clause 4: Refine Never Overwrite

**Status:** ✓ CLEAR

**Evidence:** Residual add at line 170 is `x + delta`, not in-place overwrite.

**Key:** The residual connection uses addition, not assignment. The original signal is refined, not destroyed. No `+=` on the residual tensor.

---

### Clause 5: Highway Open End-to-End

**Status:** ⚠ RISK IDENTIFIED

**Evidence:** Layer 30 residual path may drop if checkpoint omit — key `skip_residual=false` at config:44.

**Key:** The configuration contains a `skip_residual` flag. If this is ever set to `true` (or if gradient checkpointing misconfigures the residual path), the highway closes at layer 30. This is the failure mode that will burn the budget.

**Cited Location:** config.yaml line 44, `skip_residual: false`

---

## Severity Story

If residual drops at layer 30 around step 40k, loss spikes >2.0 within 2 hours and gradients explode.

**Timeline to Failure:**
- Steps 0-39k: Training appears normal
- Step ~40k: Residual path fails silently
- Within 2 hours: Loss exceeds 2.0
- Shortly after: Gradient explosion, run unrecoverable
- Result: Quarter's GPU budget burned, contract deadline missed

---

## Launch Call

**Decision:** Launch-with-conditions

**Conditions:**
1. Ada owns residual path audit by Friday
2. Hold if `skip_residual` ever true in any config variant
3. 200-step smoke run must show gradients alive at layer 30
4. Tripwire monitoring active from step 0

---

## Tripwire Protocol

**Metric:** `train/loss`

**Frequency:** Every 100 steps

**Threshold:** Stop if loss > 2.0 for 3 consecutive windows

**Owner:** Ada on-call Slack

**Escalation:** If tripwire fires, immediately:
1. Pause training
2. Snapshot checkpoint
3. Dump gradient norms for layers 28-34
4. Page Ada

---

## The Builder's Run Checklist

- [ ] Clone preserved at line 142 — verified
- [ ] Pre-norm at line 118 — verified
- [ ] FFN 4x expand at lines 155-168 — verified
- [ ] Addition not overwrite at line 170 — verified
- [ ] `skip_residual=false` at config:44 — **AUDIT REQUIRED**
- [ ] 200-step smoke run completed — pending
- [ ] Tripwire configured in monitoring — pending
- [ ] Ada assigned residual audit — pending

---

*Charter generated as part of pre-flight check. Top risk: highway_open_end_to_end.*