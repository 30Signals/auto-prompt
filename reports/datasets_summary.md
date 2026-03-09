# Dataset Labeling Summary

## Verified Gold Labels (Evidence)

### legal_contract (CUAD)

- Gold labels are present in each row under `answers.text` and `answers.answer_start`.
- Verified examples from local cached dataset:

```text
ID: LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement__Document Name
Question: Highlight the parts (if any) of this contract related to "Document Name" ...
Gold answers.text: ['SUPPLY CONTRACT']
Gold answers.answer_start: [14]

ID: LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement__Parties
Question: Highlight the parts (if any) of this contract related to "Parties" ...
Gold answers.text: ['The seller:', 'The buyer/End-User: Shenzhen LOHAS Supply Chain Management Co., Ltd.']
Gold answers.answer_start: [143, 49]

ID: LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement__Expiration Date
Question: Highlight the parts (if any) of this contract related to "Expiration Date" ...
Gold answers.text: ['The Contract is valid for 5 years, beginning from and ended on .']
Gold answers.answer_start: [10985]
```

### medical_ner (NCBI Disease)

- Gold labels are present as BIO tags under `ner_tags`, aligned with `tokens`.
- Verified examples from local cached dataset:

```text
ID: 0
Tokens: ['Clustering', 'of', 'missense', 'mutations', 'in', 'the', 'ataxia', '-', 'telangiectasia', 'gene', ...]
Gold ner_tags: ['O', 'O', 'O', 'O', 'O', 'O', 'B-Disease', 'I-Disease', 'I-Disease', 'O', ...]

ID: 1
Tokens: ['Ataxia', '-', 'telangiectasia', '(', 'A', '-', 'T', ')', 'is', 'a', ...]
Gold ner_tags: ['B-Disease', 'I-Disease', 'I-Disease', 'O', 'B-Disease', 'I-Disease', 'I-Disease', 'O', ...]

ID: 2
Tokens: ['The', 'risk', 'of', 'cancer', ',', 'especially', 'lymphoid', 'neoplasias', ...]
Gold ner_tags: ['O', 'O', 'O', 'B-Disease', 'O', 'O', 'B-Disease', 'I-Disease', ...]
```

### company_legal_risk

- Current file contains labels, but they are not independent gold truth.
- In `retrieval_replay_labeling_queue.json`, labels match `model_output` 1:1 for reviewed checks.
- This should be treated as draft/self-labeled data, not final annotated ground truth.

## Current Status by Project

- `legal_contract`: Labeled properly (gold annotations in CUAD dataset).
- `medical_ner`: Labeled properly (gold BIO tags in NCBI dataset).
- `company_legal_risk`: Not properly labeled as true ground truth yet.

## Recommended Next Action

1. Build independent human-reviewed labels for `company_legal_risk` (2 annotators + adjudication), then re-run evaluation.
