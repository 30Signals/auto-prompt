# Company Legal Risk Experiment

This experiment adds an agentic legal-risk workflow using the same optimization pattern as other tasks:
- Baseline handcrafted prompt
- DSPy student module optimized via BootstrapFewShot + COPRO
- Side-by-side evaluation and saved prompt artifacts

## Task
Input: company name + retrieved evidence snippets (lawsuits, compliance issues, news)
Output:
- `risk_level` in `{LOW, MEDIUM, HIGH, CRITICAL}`
- `should_work_with_company` in `{YES, NO}`
- summary and key findings

## Security
Use environment variables only for SearXNG access:
- `SEARXNG_BASE_URL`
- `SEARXNG_USERNAME`
- `SEARXNG_PASSWORD`

## Run Offline Benchmark
```bash
python -m experiments.company_legal_risk.run
python -m experiments.company_legal_risk.run --multi-run 5
```

## Live Agentic Data Collection (for 200-500 real cases)
1. Run live demo and log every API run:
```bash
python -m experiments.company_legal_risk.live_demo --log-file experiments/company_legal_risk/data/replay_runs.jsonl
```

Each record includes:
- timestamp
- company name
- query log
- retrieved docs
- model output

2. Build analyst labeling queue:
```bash
python -m experiments.company_legal_risk.build_labeling_queue --input-log experiments/company_legal_risk/data/replay_runs.jsonl --output-file experiments/company_legal_risk/data/retrieval_replay_labeling_queue.json --max-records 500
```

3. Analysts fill only labels:
- `risk_level`
- `should_work_with_company`
- `key_findings` (3-5)

Schema for labels:
- `experiments/company_legal_risk/data/replay_labeling_schema.json`

Pattern-based scoring schema for richer annotation:
- `experiments/company_legal_risk/data/replay_labeling_schema_v2.json`

Pattern-scoring design note:
- `experiments/company_legal_risk/PATTERN_SCORING_DATASET.md`

## Strict Agentic Evaluation (Retrieval Replay)
Use fixed retrieved evidence and gold labels.

1. Copy template:
`experiments/company_legal_risk/data/retrieval_replay_cases.template.json`

2. Fill your annotated replay cases (real API retrieval snapshots + gold labels).

3. Run:
```bash
python -m experiments.company_legal_risk.replay_eval --replay-file experiments/company_legal_risk/data/retrieval_replay_cases.template.json
```

Outputs are saved to:
`experiments/company_legal_risk/results/replay/`

## Agentic Evaluation Metrics
- `risk_exact_accuracy`: exact risk class match
- `risk_distance_score`: partial credit for near misses
- `decision_accuracy`: YES/NO correctness
- `evidence_recall`: expected risk findings recovered from rationale
- `overall_score`: weighted blend used for comparison

This supports optimization on agentic behavior because scoring rewards:
- correct decisions
- calibrated risk classification
- evidence-grounded reasoning
