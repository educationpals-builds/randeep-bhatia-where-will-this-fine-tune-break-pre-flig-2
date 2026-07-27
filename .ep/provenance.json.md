{
  "schema_version": "1.0",
  "build_name": "Where will this fine-tune break? pre-flight check for any open model you're about to train",
  "generated_at": "2024",
  "disclosure": {
    "ai_drafted": true,
    "marking": "ai_drafted",
    "eu_ai_act_article_50_compliance": true,
    "statement": "This workshop content was generated with AI assistance and is disclosed as AI-drafted material."
  },
  "field_attribution": {
    "learner_provided": [
      "specimen",
      "specimen_stakes",
      "standard_line",
      "run_reality",
      "block_findings",
      "top_risk",
      "severity_note",
      "run_call",
      "watch_tripwire"
    ],
    "ai_drafted": [
      "README.md structure and prose",
      "charter.md document organization",
      "blueprints/pre-flight-bench.md auditor specification",
      "prompts/clause-walk-pack.md prompt templates",
      "METHOD.md framework exposition",
      "VERIFY.md verification protocol",
      "provenance.json metadata structure"
    ]
  },
  "learner_field_bag": {
    "specimen": "34-layer open decoder model, block code lifted from a tutorial repo, fine-tuning on 180k clinical intake notes, launch in 9 days",
    "specimen_stakes": "A run that diverges at layer 30 burns the quarter's GPU budget and pushes the delivery date past the contract review",
    "standard_line": "Safe to train means: every one of the six operations in the block is accounted for in the code we will actually run, normalization sits before each sub-layer, the input path is preserved by addition in both halves, and a 200-step smoke run shows gradients alive in the deepest layers",
    "run_reality": "8×A100 for ~11 days, fp16 by default, 34 layers, sequence length 4096, nobody on the team has traced the forward pass end to end, and the repo's README says 'works out of the box'",
    "block_findings": {
      "clean_copy_of_the_input": "Line 142: residual = x.clone() before attn — clean copy preserved.",
      "normalization_placement": "CLEAR — because LayerNorm runs before attn at line 118 (pre-norm).",
      "where_the_work_happens": "FFN at lines 155-168 does 4x expand — work happens in MLP not attention.",
      "refine_never_overwrite": "CLEAR — because residual add at line 170 is x + delta, not in-place overwrite.",
      "highway_open_end_to_end": "Risk: layer 30 residual path may drop if checkpoint omit — key skip_residual=false at config:44."
    },
    "top_risk": "highway_open_end_to_end",
    "severity_note": "If residual drops at layer 30 around step 40k, loss spikes >2.0 within 2 hours and gradients explode.",
    "run_call": "Launch-with-conditions: Ada owns residual path audit by Friday; hold if skip_residual ever true.",
    "watch_tripwire": "Watch train/loss every 100 steps; stop if loss > 2.0 for 3 windows. Owner: Ada on-call Slack."
  },
  "source_type": "pooled",
  "license": "Educational use",
  "verification_method": "Stranger verification via VERIFY.md protocol"
}