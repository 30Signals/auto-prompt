"""
DSPy metadata pipeline for legal_contract CSV task.

Trains/evaluates directly on contract-level metadata columns using contract text
fetched from CUAD by contract_id.
"""

import argparse
import json
import random
import re
from pathlib import Path

import dspy
import pandas as pd
from datasets import load_dataset

from shared.llm_providers import setup_dspy_lm
from shared.optimization import run_two_stage_optimization
from shared.evaluation import compare_results, save_results_json, EvaluationResult
from experiments.legal_contract.metrics import llm_semantic_match_score, normalize_text
from experiments.legal_contract.metadata_normalization import (
    normalize_governing_law,
    governing_law_match_score,
    indemnification_match_score,
    expiration_match_score,
)


FIELDS = [
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Governing Law",
    "Indemnification",
    "Limitation Of Liability",
    "Non-Compete",
    "Parties",
    "Termination For Convenience",
]

FIELD_TO_ATTR = {
    "Agreement Date": "agreement_date",
    "Effective Date": "effective_date",
    "Expiration Date": "expiration_date",
    "Governing Law": "governing_law",
    "Indemnification": "indemnification",
    "Limitation Of Liability": "limitation_of_liability",
    "Non-Compete": "non_compete",
    "Parties": "parties",
    "Termination For Convenience": "termination_for_convenience",
}

# Higher weight for weak fields so optimization prioritizes them.
FIELD_WEIGHTS = {
    "Indemnification": 1.35,
    "Expiration Date": 1.30,
    "Termination For Convenience": 1.25,
}


_GOV_LAW_ALIASES = {
    "state of new york": "New York",
    "new york state": "New York",
    "laws of new york": "New York",
    "state of delaware": "Delaware",
    "laws of delaware": "Delaware",
    "commonwealth of massachusetts": "Massachusetts",
    "people's republic of china": "China",
    "prc": "China",
    "hong kong sar": "Hong Kong",
}

_GOV_LAW_CANDIDATES = [
    "england and wales", "new york", "delaware", "california", "texas", "massachusetts", "florida",
    "illinois", "new jersey", "pennsylvania", "washington", "virginia", "ohio", "michigan",
    "united states", "england", "wales", "scotland", "ireland", "united kingdom", "uk",
    "germany", "france", "spain", "italy", "switzerland", "netherlands", "canada", "australia",
    "singapore", "hong kong", "china", "japan", "india", "israel", "bermuda", "cayman islands"
]


def _canonical_title(text: str) -> str:
    return " ".join(w.capitalize() for w in str(text).split())


def _normalize_governing_law(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return "NOT FOUND"
    low = " ".join(v.lower().split())
    if low in {"not found", "none", "n/a", "na"}:
        return "NOT FOUND"

    for k, canon in _GOV_LAW_ALIASES.items():
        if k in low:
            return canon

    noise = [
        "governed by", "governing law", "laws of", "law of", "in accordance with",
        "venue", "jurisdiction", "arbitration", "exclusive", "courts of"
    ]
    cleaned = low
    for n in noise:
        cleaned = cleaned.replace(n, " ")
    cleaned = " ".join(cleaned.split())

    hits = [c for c in _GOV_LAW_CANDIDATES if re.search(rf"\b{re.escape(c)}\b", cleaned)]
    if hits:
        hits.sort(key=len, reverse=True)
        return _canonical_title(hits[0])

    if len(cleaned.split()) <= 3:
        return _canonical_title(cleaned)
    return "NOT FOUND"


def _build_contract_text_map(cache_dir: str):
    ds = load_dataset(
        "theatticusproject/cuad-qa",
        revision="refs/convert/parquet",
        cache_dir=cache_dir,
    )
    out = {}
    for split_name in ("train", "test"):
        for item in ds[split_name]:
            rid = str(item.get("id", ""))
            cid = rid.split("__", 1)[0]
            ctx = str(item.get("context", "") or "")
            if cid and ctx and cid not in out:
                out[cid] = ctx
    return out


def _safe_text(v):
    s = str(v).strip()
    return s if s else "NOT FOUND"


def _train_val_split(examples, val_ratio: float, seed: int):
    if not examples:
        return [], []
    if val_ratio <= 0:
        return examples, []

    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)

    val_size = max(1, int(round(len(items) * val_ratio)))
    # Keep at least 1 sample in train.
    if val_size >= len(items):
        val_size = len(items) - 1

    valset = items[:val_size]
    trainset = items[val_size:]
    return trainset, valset


def load_metadata_examples(csv_path: Path, text_map: dict):
    df = pd.read_csv(csv_path)
    id_col = "contract_id" if "contract_id" in df.columns else df.columns[0]
    examples = []

    for _, row in df.iterrows():
        cid = str(row[id_col]).strip()
        if not cid:
            continue
        contract_text = text_map.get(cid, "")
        if not contract_text:
            continue

        attrs = {
            "contract_id": cid,
            "contract_text": contract_text,
            "gold_metadata": {},
        }
        for field in FIELDS:
            val = _safe_text(row.get(field, "NOT FOUND"))
            attrs[FIELD_TO_ATTR[field]] = val
            attrs["gold_metadata"][field] = val

        ex = dspy.Example(**attrs).with_inputs("contract_text")
        examples.append(ex)

    return examples


class ContractMetadataSignature(dspy.Signature):
    """Extract contract metadata fields from contract text.

    Rules:
    - Return concise values only, never section numbers like "Section 19".
    - Use "NOT FOUND" only when the contract truly lacks the clause/value.
    - Expiration Date: return explicit date if present, else structured term phrase (e.g., "3-Year (36 months) Initial Term" with "Auto-Renewal" if present).
    - Indemnification: return "NOT FOUND" unless explicit indemnify/hold harmless/defend obligation exists.
    - Termination For Convenience: return only convenience/without-cause termination signal with notice if available.
    - Governing Law: return only jurisdiction name (e.g., "Delaware", "New York", "England and Wales"), not full sentence.
    """

    contract_text = dspy.InputField(desc="Full contract text")

    agreement_date = dspy.OutputField(desc="Agreement Date value or NOT FOUND")
    effective_date = dspy.OutputField(desc="Effective Date value or NOT FOUND")
    expiration_date = dspy.OutputField(desc="Expiration date or normalized term phrase with duration/renewal, else NOT FOUND")
    governing_law = dspy.OutputField(desc="Governing Law jurisdiction only (state/country), or NOT FOUND")
    indemnification = dspy.OutputField(desc="Indemnification obligation only; else NOT FOUND")
    limitation_of_liability = dspy.OutputField(desc="Limitation Of Liability value or NOT FOUND")
    non_compete = dspy.OutputField(desc="Non-Compete value or NOT FOUND")
    parties = dspy.OutputField(desc="Parties value or NOT FOUND")
    termination_for_convenience = dspy.OutputField(desc="Termination-for-convenience signal with notice/party if present, else NOT FOUND")


class BaselineMetadataModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ContractMetadataSignature)

    def forward(self, contract_text):
        return self.predictor(contract_text=contract_text)


class StudentMetadataModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(ContractMetadataSignature)

    def forward(self, contract_text):
        return self.predictor(contract_text=contract_text)


def _field_score(gold: str, pred: str, field: str, use_llm=True):
    g = _safe_text(gold)
    p = _safe_text(pred)

    if field == "Governing Law":
        rule_score = governing_law_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if field == "Indemnification":
        rule_score = indemnification_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if field == "Expiration Date":
        rule_score = expiration_match_score(p, g)
        if rule_score >= 1.0 or not use_llm:
            return rule_score
        llm_score = llm_semantic_match_score(p, [g], field) if use_llm else None
        return max(rule_score, llm_score if llm_score is not None else 0.0)

    if use_llm:
        llm_score = llm_semantic_match_score(p, [g], field)
        if llm_score is not None:
            return llm_score

    return 1.0 if normalize_text(g) == normalize_text(p) else 0.0


def validate_metadata_prediction(example, pred, trace=None):
    # Weighted lexical metric for optimization loop.
    weighted_sum = 0.0
    total_weight = 0.0
    for field in FIELDS:
        attr = FIELD_TO_ATTR[field]
        g = getattr(example, attr, "NOT FOUND")
        p = getattr(pred, attr, "NOT FOUND")
        s = 1.0 if normalize_text(g) == normalize_text(p) else 0.0
        w = FIELD_WEIGHTS.get(field, 1.0)
        weighted_sum += s * w
        total_weight += w
    return (weighted_sum / total_weight) if total_weight else 0.0


def evaluate_module(module, testset, name: str, use_llm=True):
    results = []
    per_field = {f: [] for f in FIELDS}

    for i, ex in enumerate(testset):
        try:
            pred = module(contract_text=ex.contract_text)
            row_score = []
            row_detail = {}
            for field in FIELDS:
                attr = FIELD_TO_ATTR[field]
                gold = getattr(ex, attr, "NOT FOUND")
                pv = getattr(pred, attr, "NOT FOUND")
                if field == "Governing Law":
                    gold_eval = normalize_governing_law(gold)
                    pred_eval = normalize_governing_law(pv)
                else:
                    gold_eval = gold
                    pred_eval = pv
                s = _field_score(gold_eval, pred_eval, field, use_llm=use_llm)
                per_field[field].append(s)
                row_score.append(s)
                row_detail[field] = {
                    "gold": _safe_text(gold_eval),
                    "pred": _safe_text(pred_eval),
                    "score": s,
                }

            overall = sum(row_score) / len(row_score) if row_score else 0.0
            results.append({
                "index": i,
                "contract_id": getattr(ex, "contract_id", ""),
                "details": row_detail,
                "overall_score": overall,
            })
        except Exception as e:
            results.append({
                "index": i,
                "contract_id": getattr(ex, "contract_id", ""),
                "error": str(e),
                "overall_score": 0.0,
            })
            for field in FIELDS:
                per_field[field].append(0.0)

    total = len(testset)
    per_field_avg = {
        f: (sum(vals) / len(vals) if vals else 0.0)
        for f, vals in per_field.items()
    }
    overall = sum(r["overall_score"] for r in results) / total if total else 0.0

    return EvaluationResult(
        name=name,
        results=results,
        field_accuracies={f: per_field_avg[f] for f in FIELDS},
        overall_accuracy=overall,
        total_samples=total,
        metadata={
            "per_field_accuracy": per_field_avg,
            "llm_eval_used": bool(use_llm),
        },
    )


def print_summary(result: EvaluationResult):
    print("\n" + "=" * 50)
    print(f"Evaluation Summary: {result.name}")
    print("=" * 50)
    print(f"Overall Accuracy: {result.overall_accuracy:.2%}")
    print("Per-field accuracy:")
    per = result.metadata.get("per_field_accuracy", {})
    for field, score in sorted(per.items()):
        print(f"  {field:28} {score:.2%}")


def _render_signature_prompt_text() -> str:
    sig = ContractMetadataSignature
    lines = [
        "# Baseline Metadata Prompt (Signature)",
        "",
        (sig.__doc__ or "").strip(),
        "",
        "## Fields",
    ]
    for field in FIELDS:
        lines.append(f"- {field}")

    # Best-effort append of model field metadata when available.
    model_fields = getattr(sig, "model_fields", {}) or {}
    if model_fields:
        lines.append("")
        lines.append("## Field Metadata")
        for fname, fobj in model_fields.items():
            desc = getattr(fobj, "description", "") or ""
            if desc:
                lines.append(f"- {fname}: {desc}")

    return "\n".join(lines).strip() + "\n"


def _extract_optimized_prompt_text(optimized_module: dspy.Module) -> str:
    # Best effort: COPRO may store optimized instructions internally.
    predictor = getattr(optimized_module, "predictor", None)
    parts = ["# Optimized Metadata Prompt (Best Effort)", ""]

    signature = getattr(predictor, "signature", None)
    if signature is not None:
        sig_doc = (getattr(signature, "__doc__", "") or "").strip()
        if sig_doc:
            parts.append(sig_doc)
            parts.append("")

    for attr in ("instructions", "_instructions", "optimized_prompt"):
        val = getattr(predictor, attr, None)
        if isinstance(val, str) and val.strip():
            parts.append(f"## {attr}")
            parts.append(val.strip())
            parts.append("")

    if len(parts) <= 2:
        parts.append("No explicit optimized instruction string exposed by DSPy object.")
        parts.append("Using baseline signature prompt as closest reference.")
        parts.append("")
        parts.append(_render_signature_prompt_text())

    return "\n".join(parts).strip() + "\n"


def _save_prompt_artifacts(output_dir: Path, baseline_module: dspy.Module, optimized_module: dspy.Module):
    baseline_prompt = _render_signature_prompt_text()
    optimized_prompt = _extract_optimized_prompt_text(optimized_module)

    baseline_path = output_dir / "baseline_prompt.txt"
    optimized_path = output_dir / "optimized_prompt.txt"
    compare_path = output_dir / "prompt_comparison.md"

    baseline_path.write_text(baseline_prompt, encoding="utf-8")
    optimized_path.write_text(optimized_prompt, encoding="utf-8")

    comparison = (
        "# Prompt Comparison\n\n"
        "## Baseline Prompt\n\n"
        f"```text\n{baseline_prompt}\n```\n\n"
        "## Optimized Prompt\n\n"
        f"```text\n{optimized_prompt}\n```\n"
    )
    compare_path.write_text(comparison, encoding="utf-8")

def _save_runtime_prompt_history(output_dir: Path):
    """Save actual LM prompt history captured during the run."""
    lm = dspy.settings.lm
    history = getattr(lm, "history", None)
    if not isinstance(history, list) or not history:
        return

    raw_path = output_dir / "dspy_runtime_history.json"
    raw_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Best-effort plain text extraction of prompt/messages per call.
    prompts_txt = []
    for i, item in enumerate(history, start=1):
        prompts_txt.append(f"===== CALL {i} =====")
        if isinstance(item, dict):
            if item.get("prompt"):
                prompts_txt.append("[prompt]")
                prompts_txt.append(str(item.get("prompt")))
            if item.get("messages"):
                prompts_txt.append("[messages]")
                prompts_txt.append(json.dumps(item.get("messages"), ensure_ascii=False, indent=2, default=str))
            if item.get("response"):
                prompts_txt.append("[response]")
                prompts_txt.append(str(item.get("response")))
        else:
            prompts_txt.append(str(item))
        prompts_txt.append("")

    txt_path = output_dir / "dspy_runtime_prompts.txt"
    txt_path.write_text("\n".join(prompts_txt), encoding="utf-8")

def run_metadata_dspy(
    train_csv: Path,
    test_csv: Path,
    cache_dir: str,
    output_dir: Path,
    use_llm_eval: bool,
    val_ratio: float,
    seed: int,
):
    print("=" * 70)
    print("LEGAL CONTRACT METADATA DSPY PIPELINE")
    print("=" * 70)

    print("\n[1/5] Setting up LLM...")
    setup_dspy_lm()

    print("[2/5] Loading metadata datasets + CUAD contract text...")
    text_map = _build_contract_text_map(cache_dir)
    full_trainset = load_metadata_examples(train_csv, text_map)
    testset = load_metadata_examples(test_csv, text_map)
    if not full_trainset or not testset:
        raise ValueError("Empty train/test set after loading metadata + contract text map")

    opt_trainset, valset = _train_val_split(full_trainset, val_ratio=val_ratio, seed=seed)
    print(f"      Train (input): {len(full_trainset)} contracts, Test: {len(testset)} contracts")
    print(f"      Optimization split: train={len(opt_trainset)}, validation={len(valset)}")

    print("[3/5] Evaluating baseline...")
    baseline = BaselineMetadataModule()
    baseline_results = evaluate_module(baseline, testset, "BaselineMetadata", use_llm=use_llm_eval)
    print_summary(baseline_results)

    # Validation anchor for safer model selection.
    baseline_val = evaluate_module(baseline, valset, "BaselineValidation", use_llm=False) if valset else None

    print("[4/5] Optimizing with DSPy...")
    student = StudentMetadataModule()
    optimized = run_two_stage_optimization(
        student_module=student,
        trainset=opt_trainset,
        metric=validate_metadata_prediction,
        bootstrap_config={
            "max_bootstrapped_demos": 6,
            "max_labeled_demos": 6,
            "max_rounds": 2,
            "max_errors": 4,
        },
        copro_config={
            "breadth": 3,
            "depth": 1,
            "init_temperature": 0.2,
        },
    )

    # Keep optimized model only if it beats baseline on lexical validation.
    if valset:
        optimized_val = evaluate_module(optimized, valset, "DSPyValidation", use_llm=False)
        print(f"      Validation baseline: {baseline_val.overall_accuracy:.2%}")
        print(f"      Validation DSPy:     {optimized_val.overall_accuracy:.2%}")
        if optimized_val.overall_accuracy < baseline_val.overall_accuracy:
            print("      DSPy underperformed on validation. Falling back to baseline for final test evaluation.")
            final_module = baseline
            final_name = "DSPyMetadata(FallbackBaseline)"
        else:
            final_module = optimized
            final_name = "DSPyMetadata"
    else:
        final_module = optimized
        final_name = "DSPyMetadata"

    print("[5/5] Evaluating optimized...")
    optimized_results = evaluate_module(final_module, testset, final_name, use_llm=use_llm_eval)
    print_summary(optimized_results)

    comparison = compare_results(baseline_results, optimized_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_results_json(baseline_results, output_dir / "baseline_metadata_results.json")
    save_results_json(optimized_results, output_dir / "dspy_metadata_results.json")
    save_results_json(comparison, output_dir / "comparison_metadata_results.json")
    _save_prompt_artifacts(output_dir, baseline, final_module)
    _save_runtime_prompt_history(output_dir)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Baseline overall:  {baseline_results.overall_accuracy:.2%}")
    print(f"DSPy overall:      {optimized_results.overall_accuracy:.2%}")
    print(f"Improvement:       {(optimized_results.overall_accuracy - baseline_results.overall_accuracy):+.2%}")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run DSPy metadata pipeline on contract metadata CSV files")
    parser.add_argument(
        "--train-csv",
        default="experiments/legal_contract/data/reviewed/contract_metadata_cleaned_train.csv",
        help="Train metadata CSV",
    )
    parser.add_argument(
        "--test-csv",
        default="experiments/legal_contract/data/reviewed/contract_metadata_cleaned_test.csv",
        help="Test metadata CSV",
    )
    parser.add_argument(
        "--cache-dir",
        default="experiments/legal_contract/data/cache",
        help="HF datasets cache dir",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/legal_contract/results/metadata_dspy",
        help="Output directory",
    )
    parser.add_argument(
        "--no-llm-eval",
        action="store_true",
        help="Use lexical-only evaluation (faster)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio from training set for safer model selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split",
    )
    args = parser.parse_args()

    run_metadata_dspy(
        train_csv=Path(args.train_csv),
        test_csv=Path(args.test_csv),
        cache_dir=args.cache_dir,
        output_dir=Path(args.output_dir),
        use_llm_eval=not args.no_llm_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
