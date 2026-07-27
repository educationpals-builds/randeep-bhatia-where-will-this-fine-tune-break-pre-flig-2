# Clause Walk Pack: Five Standalone Prompts

Each prompt below examines one clause of the pre-flight check. Paste your block code or config excerpt, and the prompt returns a finding-or-earned-clear.

---

## Prompt 1: Clean Copy of the Input

```
You are checking transformer block code for the "clean copy of the input" clause.

## What You're Looking For

Before any transformation (attention, FFN), the input tensor must be preserved as a clean copy. This is typically done with:
- `residual = x.clone()`
- `residual = x.detach().clone()` (if gradient isolation needed)
- `residual = x + 0` (rare but valid)

## Red Flags
- No explicit copy before attention
- `residual = x` (reference, not copy)
- In-place operations on x before residual is saved

## Your Task

Examine the code provided and return exactly one of:

**CLEAR:** "Line [N]: [code snippet] — clean copy preserved because [reason]."

**RISK:** "Line [N]: [code snippet] — risk because [reason]. The input may be mutated before residual addition."

## Code to Examine

[USER PASTES BLOCK CODE HERE]
```

---

## Prompt 2: Normalization Placement

```
You are checking transformer block code for the "normalization placement" clause.

## What You're Looking For

In modern deep transformers (>24 layers), normalization should be PRE-NORM:
- LayerNorm BEFORE attention
- LayerNorm BEFORE FFN

Post-norm (LayerNorm after attention/FFN) causes gradient instability in deep networks.

## Pattern Recognition

Pre-norm (GOOD for deep models):
```python
x = self.norm1(x)
x = self.attn(x)
x = residual + x
```

Post-norm (RISKY for deep models):
```python
x = self.attn(x)
x = self.norm1(residual + x)
```

## Your Task

Examine the code provided and return exactly one of:

**CLEAR:** "CLEAR — because LayerNorm runs before [attn/FFN] at line [N] (pre-norm)."

**RISK:** "RISK — because LayerNorm runs after [attn/FFN] at line [N] (post-norm). For [depth] layers, this may cause gradient instability."

## Code to Examine

[USER PASTES BLOCK CODE HERE]
```

---

## Prompt 3: Where the Work Happens

```
You are checking transformer block code for the "where the work happens" clause.

## What You're Looking For

In a transformer block:
- Attention ROUTES information between positions
- FFN (MLP) COMPUTES new representations

The FFN should have:
- An expansion factor (typically 4x the model dimension)
- A nonlinearity (GELU, ReLU, SiLU)
- A projection back to model dimension

## Red Flags
- FFN with no expansion (just linear layers)
- Missing nonlinearity
- Expansion factor != 4x without explicit justification
- Heavy computation in attention (should be lightweight)

## Your Task

Examine the code provided and return:

**CLEAR:** "[Component] at lines [X-Y] does [expansion factor] expand — work happens in [MLP/attention] [assessment]."

**RISK:** "[Component] at lines [X-Y] — risk because [missing expansion / wrong location / missing nonlinearity]."

## Code to Examine

[USER PASTES BLOCK CODE HERE]
```

---

## Prompt 4: Refine Never Overwrite

```
You are checking transformer block code for the "refine never overwrite" clause.

## What You're Looking For

Residual connections must ADD, not OVERWRITE:

CORRECT:
```python
x = residual + attn_output  # Addition: refine
output = residual + ffn_output  # Addition: refine
```

INCORRECT:
```python
x += attn_output  # In-place: may corrupt
x = attn_output  # Overwrite: destroys residual
residual = attn_output  # Reassignment: loses original
```

## Your Task

Examine the code provided and return exactly one of:

**CLEAR:** "CLEAR — because residual add at line [N] is [code snippet], not in-place overwrite."

**RISK:** "RISK — because line [N] uses [+=, assignment, etc.] which [overwrites/mutates] the residual path."

## Code to Examine

[USER PASTES BLOCK CODE HERE]
```

---

## Prompt 5: Highway Open End-to-End

```
You are checking transformer configuration for the "highway open end-to-end" clause.

## What You're Looking For

The residual "highway" must remain open from input to output across ALL layers. Check for:

1. **Config flags that close the highway:**
   - `skip_residual: true`
   - `residual_dropout: 1.0`
   - `use_residual: false`

2. **Gradient checkpointing issues:**
   - Checkpointing that doesn't preserve residual path
   - Layer drops during checkpointing

3. **Conditional residuals:**
   - `if layer_idx > N: skip residual`
   - Residual scaling that goes to zero

## Your Task

Examine the config/code provided and return:

**CLEAR:** "CLEAR — residual path confirmed open. Key [config_key]=[value] at [location] preserves highway."

**RISK:** "Risk: [specific layer/condition] residual path may drop if [condition] — key [config_key]=[value] at [location]."

## Config/Code to Examine

[USER PASTES CONFIG OR CODE HERE]
```

---

## Usage Notes

1. **One prompt per clause** — don't combine them
2. **Paste actual code** — these prompts need real line numbers
3. **Include context** — mention model depth if checking normalization
4. **Collect all five** — transfer findings to charter.md
5. **Top risk** — whichever clause returns RISK with highest severity is your top_risk

---

## Quick Reference: What Each Clause Prevents

| Clause | Failure Mode | Observable Symptom |
|--------|--------------|--------------------|
| Clean copy | Corrupted residuals | NaN in early layers |
| Normalization | Gradient instability | Loss oscillation |
| Work location | Capacity mismatch | Underfitting |
| Refine not overwrite | Signal destruction | Gradient vanishing |
| Highway open | Deep layer death | Late-training collapse |