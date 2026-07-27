# Where Will This Fine-Tune Break? — Pre-Flight Check for Open Models

## The Specimen

34-layer open decoder model, block code lifted from a tutorial repo, fine-tuning on 180k clinical intake notes, launch in 9 days.

**Stakes:** A run that diverges at layer 30 burns the quarter's GPU budget and pushes the delivery date past the contract review.

## The Verdict

**Launch-with-conditions:** Ada owns residual path audit by Friday; hold if `skip_residual` ever true.

### Block Findings Summary

| Clause | Status | Evidence |
|--------|--------|----------|
| Clean copy of the input | ✓ CLEAR | Line 142: `residual = x.clone()` before attn |
| Normalization placement | ✓ CLEAR | LayerNorm runs before attn at line 118 (pre-norm) |
| Where the work happens | ✓ CLEAR | FFN at lines 155-168 does 4x expand |
| Refine never overwrite | ✓ CLEAR | Residual add at line 170 is `x + delta`, not in-place |
| Highway open end-to-end | ⚠ RISK | Layer 30 residual path may drop if checkpoint omit — key `skip_residual=false` at config:44 |

**Top Risk:** `highway_open_end_to_end`

## The Tripwire

Watch `train/loss` every 100 steps; stop if loss > 2.0 for 3 consecutive windows.

**Owner:** Ada on-call Slack.

**Severity:** If residual drops at layer 30 around step 40k, loss spikes >2.0 within 2 hours and gradients explode.

## One-Paste Rebuild Block

```bash
# Clone and enter
git clone <this-repo-url> && cd pre-flight-check

# Run the clause walk on your block code
cat your_model_block.py | python -c "
import sys
code = sys.stdin.read()
print('Paste this code into the clause-walk-pack prompts in prompts/')
print('Check each clause. Document findings in charter.md.')
print('Set tripwire before launch.')
"

# Smoke test: 200 steps, watch deepest layer gradients
python train.py --max_steps=200 --log_grad_norm=true --watch_layer=30
```

## Files

- `charter.md` — Full pre-flight findings document
- `blueprints/pre-flight-bench.md` — Conversational auditor specification
- `prompts/clause-walk-pack.md` — Five standalone prompts for clause checking
- `METHOD.md` — The BLOCK framework
- `VERIFY.md` — Stranger verification protocol
- `.ep/provenance.json` — Build provenance and AI disclosure

---

*This workshop is ai_drafted with learner-provided specimen data. See `.ep/provenance.json` for full disclosure.*

<!-- educationpals-build-verified -->