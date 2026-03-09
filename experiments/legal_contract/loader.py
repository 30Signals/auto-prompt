"""
Data Loading for Legal Contract Analysis Experiment

Downloads and parses the CUAD dataset from Hugging Face.
Caches locally to avoid repeated downloads.
"""

import json
import random
import re

import dspy
from pathlib import Path
from . import config

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Local cache directory
CACHE_DIR = Path(__file__).parent / "data" / "cache"


def download_cuad_dataset():
    """
    Download CUAD dataset from Hugging Face.
    Uses local cache to avoid repeated downloads.

    Returns:
        Dataset with train and test splits
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "Hugging Face datasets library not installed. "
            "Run: pip install datasets"
        )

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load parquet-converted dataset (compatible with modern datasets versions)
    dataset = load_dataset(
        "theatticusproject/cuad-qa",
        revision="refs/convert/parquet",
        cache_dir=str(CACHE_DIR),
    )
    return dataset


def normalize_clause_type(question):
    """
    Extract clause type from CUAD question format.

    CUAD questions follow pattern: "Highlight the parts (if any) of this contract related to [Clause Type]..."
    """
    # Parse the explicit clause label CUAD places in quotes:
    # ... related to "Clause Label" ...
    text = str(question or "")
    match = re.search(r'related to\s+"([^"]+)"', text, flags=re.IGNORECASE)
    raw_label = match.group(1).strip().lower() if match else text.strip().lower()

    # Map CUAD-native labels to our normalized taxonomy.
    clause_mappings = {
        "parties": "Parties",
        "agreement date": "Agreement Date",
        "effective date": "Effective Date",
        "expiration date": "Expiration Date",
        "governing law": "Governing Law",
        "termination for convenience": "Termination For Convenience",
        "limitation of liability": "Limitation Of Liability",
        "cap on liability": "Limitation Of Liability",
        "uncapped liability": "Limitation Of Liability",
        "indemnification": "Indemnification",
        "covenant not to sue": "Indemnification",
        "non-compete": "Non-Compete",
        "competitive restriction exception": "Non-Compete",
        "no-solicit of employees": "Non-Compete",
        "no-solicit of customers": "Non-Compete",
        "confidentiality": "Confidentiality",
        "non-disclosure": "Confidentiality",
        "nondisclosure": "Confidentiality",
        "confidential information": "Confidentiality",
        "non-disparagement": "Confidentiality",
    }

    return clause_mappings.get(raw_label)


def _compose_gold_clause_text(answer_texts):
    if not answer_texts:
        return "NOT FOUND"
    # Preserve answer order while removing exact duplicates.
    deduped = []
    seen = set()
    for ans in answer_texts:
        key = ans.strip()
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return "\n".join(deduped) if deduped else "NOT FOUND"


def _build_example(contract_text, clause_type, answer_texts):
    context = str(contract_text or "")
    answers = [str(a).strip() for a in (answer_texts or []) if str(a).strip()]
    clause_text = _compose_gold_clause_text(answers)

    # Truncate context for model limits
    if len(context) > 4000:
        context = context[:4000] + "..."

    return dspy.Example(
        contract_text=context,
        clause_type=clause_type,
        clause_text=clause_text,
        gold_clauses=answers if answers else ["NOT FOUND"],
    ).with_inputs("contract_text", "clause_type")


def _load_reviewed_examples(
    reviewed_file,
    clause_types,
    seed=None,
    allow_source_gold=False,
):
    """
    Load reviewed rows from JSONL.
    Uses reviewed_gold if present; falls back to source_gold when status is approved.
    """
    path = Path(reviewed_file)
    if not path.exists():
        raise FileNotFoundError(f"Reviewed file not found: {path}")

    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            clause_type = str(row.get("clause_type", "")).strip()
            if clause_type not in clause_types:
                continue

            status = str(row.get("review_status", "")).strip().lower()
            reviewed_gold = row.get("reviewed_gold") or []
            source_gold = row.get("source_gold") or []

            if isinstance(reviewed_gold, str):
                reviewed_gold = [reviewed_gold]
            if isinstance(source_gold, str):
                source_gold = [source_gold]

            # Keep only human-ready rows: reviewed_gold rows, and optionally
            # source_gold rows when bootstrap mode is enabled.
            if reviewed_gold:
                gold = reviewed_gold
            elif status == "approved" or allow_source_gold:
                gold = source_gold
            else:
                continue

            contract_text = row.get("contract_text", "")
            ex = _build_example(contract_text, clause_type, gold)
            examples.append(ex)

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(examples)

    return examples


def load_reviewed_examples_file(
    reviewed_file,
    clause_types,
    allow_source_gold=False,
    seed=None,
):
    """
    Load examples from a single reviewed JSONL file without re-splitting.
    """
    return _load_reviewed_examples(
        reviewed_file=reviewed_file,
        clause_types=clause_types,
        seed=seed,
        allow_source_gold=allow_source_gold,
    )


def load_data(
    train_size=None,
    test_size=None,
    seed=None,
    clause_types=None,
    reviewed_file=None,
    allow_source_gold=False,
):
    """
    Load CUAD dataset and return train/test splits as dspy.Example objects.

    Each example contains:
    - contract_text: The contract context
    - clause_type: Type of clause to extract
    - clause_text: Ground truth extracted clause (or "NOT FOUND")

    Args:
        train_size: Number of training samples. Default: config.TRAIN_SIZE
        test_size: Number of test samples. Default: config.TEST_SIZE
        seed: Random seed for shuffling. Default: None
        clause_types: List of clause types to include. Default: config.CLAUSE_TYPES

    Returns:
        Tuple of (trainset, testset) as lists of dspy.Example objects
    """
    train_size = train_size or config.TRAIN_SIZE
    test_size = test_size or config.TEST_SIZE
    clause_types = clause_types or config.CLAUSE_TYPES

    if reviewed_file:
        reviewed_examples = _load_reviewed_examples(
            reviewed_file=reviewed_file,
            clause_types=clause_types,
            seed=seed,
            allow_source_gold=allow_source_gold,
        )
        if len(reviewed_examples) < 2:
            raise ValueError(
                f"Not enough reviewed examples in {reviewed_file}. "
                "Need at least 2 reviewed/approved rows (or pass allow_source_gold=True)."
            )

        # Build a proportional split from reviewed examples.
        # This prevents degenerate 199/1 splits when train_size and test_size
        # are both large relative to reviewed set size.
        desired_train = max(1, int(train_size))
        desired_test = max(1, int(test_size))
        desired_total = desired_train + desired_test

        if desired_total < 2:
            desired_total = 2

        subset_n = min(len(reviewed_examples), desired_total)
        subset = reviewed_examples[:subset_n]

        train_ratio = desired_train / desired_total
        train_n = int(round(subset_n * train_ratio))
        train_n = max(1, min(subset_n - 1, train_n))
        test_n = subset_n - train_n

        trainset = subset[:train_n]
        testset = subset[train_n: train_n + test_n]
        return trainset, testset

    # Download dataset
    dataset = download_cuad_dataset()

    # Get train and test splits
    train_data = dataset["train"]
    test_data = dataset.get("test", dataset.get("validation", train_data))

    def build_example(item):
        # Parse question to get clause type
        question = item.get("question", "")
        clause_type = normalize_clause_type(question)

        # Skip if not a clause type we care about
        if clause_type not in clause_types:
            return None

        contract_text = item.get("context", "")
        if not contract_text:
            return None

        # Get all non-empty answer spans. CUAD can have multiple valid spans.
        answers = item.get("answers", {})
        raw_answer_texts = answers.get("text", [])
        answer_texts = [str(a).strip() for a in raw_answer_texts if str(a).strip()]
        return _build_example(contract_text, clause_type, answer_texts)

    def convert_to_examples(data, max_samples):
        # Stratified sampling by clause type to avoid domination by frequent types.
        if not clause_types:
            return []

        by_type = {ct: [] for ct in clause_types}
        seen = set()

        # Collect de-duplicated candidates for each clause type.
        for item in data:
            ex = build_example(item)
            if ex is None:
                continue

            key = (ex.clause_type, ex.contract_text[:160], ex.clause_text[:160])
            if key in seen:
                continue

            by_type[ex.clause_type].append(ex)
            seen.add(key)

        per_type_quota = max_samples // len(clause_types)
        examples = []
        indices = {ct: 0 for ct in clause_types}

        # Pass 1: take an equal base quota from each clause type.
        for ct in clause_types:
            take_n = min(per_type_quota, len(by_type[ct]))
            examples.extend(by_type[ct][:take_n])
            indices[ct] = take_n

        # Pass 2: round-robin top-up from remaining items to keep balance.
        while len(examples) < max_samples:
            added = False
            for ct in clause_types:
                idx = indices[ct]
                if idx < len(by_type[ct]):
                    examples.append(by_type[ct][idx])
                    indices[ct] += 1
                    added = True
                    if len(examples) >= max_samples:
                        break
            if not added:
                break

        return examples[:max_samples]

    # Shuffle if seed provided
    if seed is not None:
        train_data = train_data.shuffle(seed=seed)
        test_data = test_data.shuffle(seed=seed)

    trainset = convert_to_examples(train_data, train_size)
    testset = convert_to_examples(test_data, test_size)

    return trainset, testset


if __name__ == "__main__":
    try:
        train, test = load_data()
        print("Successfully loaded CUAD dataset.")
        print(f"Train size: {len(train)}")
        print(f"Test size: {len(test)}")
        print("\nSample Train Item:")
        print(f"Clause Type: {train[0].clause_type}")
        print(f"Contract: {train[0].contract_text[:200]}...")
        print(f"Answer: {train[0].clause_text[:100]}...")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
