# Pattern-Based Company Risk Dataset

The current `company_legal_risk` dataset stores only:
- `risk_level`
- `should_work_with_company`
- `key_findings`

That is enough for a coarse recommendation, but it is too compressed for:
- learning reusable risk patterns
- explaining why two `HIGH` companies are different
- producing a calibrated score out of `1.0`
- analyzing field-wise accuracy the way `legal_contract` does by clause type

## New idea

Treat legal-risk patterns like contract clause types.

Instead of one final label only, each company case can be annotated across these dimensions:
- `regulatory_enforcement`
- `litigation_exposure`
- `fraud_corruption`
- `data_privacy_cybersecurity`
- `labor_employment`
- `environmental_safety`
- `sanctions_trade`
- `governance_ethics`

Each dimension gets:
- `severity`: `0..3`
- `confidence`: `0.0..1.0`
- `evidence_refs`: which retrieved snippets support it
- `notes`: short reviewer explanation

## Severity scale

- `0`: no meaningful risk signal
- `1`: minor or isolated issue
- `2`: meaningful recurring or material issue
- `3`: severe, systemic, or government-enforcement-grade issue

## Score design

For each dimension:

`dimension_score = weight * (severity / 3.0) * confidence`

Overall pattern risk:

`pattern_risk_score = sum(dimension_score) / sum(weights for labeled dimensions)`

Recommended output fields:
- `risk_score`: normalized `0.0..1.0`
- `assessment_confidence`: normalized `0.0..1.0`
- `derived_risk_level`
- `derived_recommendation`

Suggested decision thresholds:
- `< 0.25`: `WORK`
- `< 0.50`: `WORK_WITH_GUARDRAILS`
- `< 0.75`: `ESCALATE`
- `>= 0.75`: `DO_NOT_WORK`

## Why this matches `legal_contract`

`legal_contract` improved by splitting extraction into distinct clause/metadata fields.
We can do the same here:
- each risk dimension becomes a measurable field
- we can report per-dimension accuracy
- we can see where the model is weak
- we can optimize prompts against those weak dimensions instead of only one binary recommendation

## Backward compatibility

The old labels still stay useful:
- `risk_level`
- `should_work_with_company`
- `key_findings`

They remain the top-level business output, while the pattern labels become the training and scoring backbone.

## Immediate workflow

1. Create a pilot annotation set:
`python -m experiments.company_legal_risk.prepare_pattern_labeling_pilot --input-json experiments/company_legal_risk/data/v3/retrieval_replay_labeling_queue_v3.json --output-json experiments/company_legal_risk/data/v3/pattern_labeling_pilot.json --report-json experiments/company_legal_risk/data/v3/pattern_labeling_pilot_report.json --pilot-size 40`

2. Annotate `labels.risk_dimensions` for those pilot cases.

3. Check label completeness:
`python -m experiments.company_legal_risk.evaluate_pattern_labels --input-json experiments/company_legal_risk/data/v3/pattern_labeling_pilot.json --output-json experiments/company_legal_risk/data/v3/pattern_labeling_pilot_eval.json`

4. After model predictions include pattern dimensions, run the same evaluator with `--predicted-field`.
