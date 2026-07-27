# Stranger Verification Protocol

## Purpose

This document enables anyone unfamiliar with the project to verify that the pre-flight check tooling works as specified.

---

## Verification Steps

### Step 1: Obtain the Seeded Specimen

Use this synthetic transformer block code that contains a known normalization placement issue:

```python
# Seeded specimen: TransformerBlock with POST-NORM (intentional issue)
class TransformerBlock(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, d_ff=4096):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # Line 110: Save residual
        residual = x.clone()
        
        # Line 113: Attention (no pre-norm!)
        attn_out = self.attn(x, x, x, mask)
        attn_out = self.dropout(attn_out)
        
        # Line 117: Post-norm after attention (THIS IS THE ISSUE)
        x = self.norm1(residual + attn_out)
        
        # Line 120: Save new residual
        residual = x.clone()
        
        # Line 123: FFN (no pre-norm!)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        
        # Line 127: Post-norm after FFN (ALSO AN ISSUE)
        x = self.norm2(residual + ffn_out)
        
        return x
```

### Step 2: Run the Normalization Placement Prompt

Open any capable chat interface and paste the prompt from `prompts/clause-walk-pack.md` (Prompt 2: Normalization Placement).

Then paste the seeded specimen code above.

### Step 3: Verify the Expected Finding

The tool should return a **RISK** finding that:

1. **Identifies post-norm placement** — The response should note that LayerNorm runs AFTER attention/FFN, not before

2. **Cites specific lines** — Should reference:
   - Line 117: `self.norm1(residual + attn_out)` — norm after attention
   - Line 127: `self.norm2(residual + ffn_out)` — norm after FFN

3. **Explains the risk** — Should mention gradient instability in deep networks

### Expected Output Pattern

```
RISK — because LayerNorm runs after attention at line 117 (post-norm). 
For 34 layers, this may cause gradient instability. 
Also: LayerNorm runs after FFN at line 127 (post-norm).
```

### Step 4: Confirm Finding Accuracy

Verify that:
- [ ] The finding correctly identifies POST-NORM architecture
- [ ] Line 117 is cited (or approximate, given model variations)
- [ ] Line 127 is cited (or approximate)
- [ ] The risk explanation mentions deep networks or gradient instability
- [ ] The finding does NOT incorrectly say "CLEAR"

---

## Verification Checklist

| Check | Expected | Actual | Pass? |
|-------|----------|--------|-------|
| Identifies post-norm | Yes | | |
| Cites attention norm line | ~117 | | |
| Cites FFN norm line | ~127 | | |
| Mentions gradient risk | Yes | | |
| Does not false-clear | Correct | | |

---

## Troubleshooting

**If the tool returns CLEAR:**
The prompt may not be correctly loaded. Ensure you copied the entire prompt including the "What You're Looking For" section.

**If line numbers differ:**
Minor variations (±2 lines) are acceptable. The key is that the post-norm pattern is identified.

**If no risk explanation:**
The model may need the context that this is a 34-layer network. Add: "This is for a 34-layer model" after pasting the code.

---

## Secondary Verification: Clean Copy Clause

To verify a second clause, use the same specimen with Prompt 1 (Clean Copy of the Input).

**Expected result:** CLEAR — because line 110 shows `residual = x.clone()`

This confirms the tool correctly identifies both passing and failing clauses.

---

## Verification Complete When

1. Normalization placement prompt returns RISK with line citations
2. Clean copy prompt returns CLEAR with line citation
3. Both findings match the actual code structure

If all three conditions are met, the pre-flight check tooling is verified functional.