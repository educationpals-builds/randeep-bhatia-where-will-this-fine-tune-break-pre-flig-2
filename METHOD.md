# METHOD: The BLOCK Framework

## Framework Name: BLOCK

**B**efore-transform copy preserved  
**L**ayerNorm placement verified  
**O**peration location confirmed  
**C**ombine by addition, not overwrite  
**K**eep highway open end-to-end  

---

## The Five Clauses Expanded

### B — Before-Transform Copy Preserved

**Question:** Is the input tensor cloned before any transformation touches it?

**Why it matters:** Attention and FFN operations may mutate tensors in-place. If the residual reference points to the same memory, the "skip connection" carries corrupted data.

**What to check:**
- Explicit `.clone()` call before attention
- No in-place operations between input and residual save
- Residual variable is not just a reference

**Pass condition:** Line number where clean copy is created, with code snippet.

---

### L — LayerNorm Placement Verified

**Question:** Does normalization happen BEFORE each sub-layer (pre-norm)?

**Why it matters:** Post-norm architectures suffer gradient degradation in deep networks. Pre-norm keeps gradients healthy through 30+ layers.

**What to check:**
- LayerNorm call appears before `self.attn()`
- LayerNorm call appears before `self.ffn()`
- No normalization AFTER the residual add

**Pass condition:** "Pre-norm confirmed" with line numbers for both norm calls.

---

### O — Operation Location Confirmed

**Question:** Is the heavy computation happening in the FFN, not attention?

**Why it matters:** Attention is O(n²) in sequence length but should be lightweight per-position. FFN does the actual representation learning with its 4x expansion.

**What to check:**
- FFN has expansion factor (typically 4x model dim)
- FFN has nonlinearity (GELU, SiLU, ReLU)
- Attention doesn't contain extra linear layers

**Pass condition:** FFN expansion ratio and nonlinearity identified with line numbers.

---

### C — Combine by Addition, Not Overwrite

**Question:** Does the residual connection use `+` rather than `=` or `+=`?

**Why it matters:** The residual stream must be REFINED, not REPLACED. Overwriting destroys the gradient highway.

**What to check:**
- `output = residual + delta` pattern
- No `residual = new_value` after initial save
- No `tensor += value` on residual path

**Pass condition:** Addition operation identified with line number, confirmed not in-place.

---

### K — Keep Highway Open End-to-End

**Question:** Can gradients flow from output to input through every layer's residual path?

**Why it matters:** If any layer closes the highway (via config flag, dropout=1.0, or conditional skip), gradients die at that layer. Deep layers stop learning.

**What to check:**
- `skip_residual` or similar flags are `false`
- Residual dropout < 1.0
- No conditional residual skipping based on layer index
- Gradient checkpointing preserves residual path

**Pass condition:** Config key and value that confirms highway is open, with location.

---

## Applying BLOCK

### Step 1: Gather Materials
- Transformer block source code (forward pass)
- Model configuration file
- Training configuration (for checkpoint settings)

### Step 2: Walk Each Letter
Use the prompts in `prompts/clause-walk-pack.md` to check each clause.

### Step 3: Document Findings
Record in `charter.md`:
- Status (CLEAR or RISK)
- Evidence (line number, code snippet)
- Key (config value if applicable)

### Step 4: Identify Top Risk
The clause with RISK status and highest severity becomes `top_risk`.

### Step 5: Set Tripwire
Based on the failure mode of your top risk, configure monitoring.

### Step 6: Make Launch Call
- All CLEAR → Launch (with smoke run)
- Any RISK with mitigation → Launch-with-conditions
- Any RISK without mitigation → Hold

---

## Why BLOCK Works

The framework targets the five ways a transformer block can silently fail during fine-tuning:

1. **Corrupted residuals** → Caught by B
2. **Gradient instability** → Caught by L
3. **Capacity mismatch** → Caught by O
4. **Signal destruction** → Caught by C
5. **Highway closure** → Caught by K

Each failure mode has a different timeline and symptom. BLOCK ensures you've checked for all five before committing compute.

---

## Framework Lineage

BLOCK synthesizes insights from:
- Residual network theory (He et al.)
- Pre-norm transformer analysis (Xiong et al.)
- Gradient flow studies in deep networks
- Production fine-tuning failure post-mortems

The letters appear only in this document. All other files reference clauses by their full names.