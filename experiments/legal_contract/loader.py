"""
Data Loading for Legal Contract Analysis Experiment

Downloads and parses the CUAD dataset from Hugging Face.
Caches locally to avoid repeated downloads.
"""

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
    
    # Load the CUAD QA dataset with local cache
    dataset = load_dataset("theatticusproject/cuad-qa", cache_dir=str(CACHE_DIR))
    return dataset


def normalize_clause_type(question):
    """
    Extract clause type from CUAD question format.
    
    CUAD questions follow pattern: "Highlight the parts (if any) of this contract related to [Clause Type]..."
    """
    # Map question patterns to clause types
    clause_mappings = {
        "parties": "Parties",
        "agreement date": "Agreement Date",
        "effective date": "Effective Date",
        "expiration date": "Expiration Date",
        "governing law": "Governing Law",
        "termination for convenience": "Termination For Convenience",
        "limitation of liability": "Limitation Of Liability",
        "indemnification": "Indemnification",
        "non-compete": "Non-Compete",
        "confidentiality": "Confidentiality",
    }
    
    question_lower = question.lower()
    for key, clause_type in clause_mappings.items():
        if key in question_lower:
            return clause_type
    
    return None


def load_data(train_size=None, test_size=None, seed=None, clause_types=None):
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
    
    # Download dataset
    dataset = download_cuad_dataset()
    
    # Get train and test splits
    train_data = dataset['train']
    test_data = dataset.get('test', dataset.get('validation', train_data))
    
    def convert_to_examples(data, max_samples):
        examples = []
        
        for item in data:
            if len(examples) >= max_samples:
                break
            
            # Parse question to get clause type
            question = item.get('question', '')
            clause_type = normalize_clause_type(question)
            
            # Skip if not a clause type we care about
            if clause_type not in clause_types:
                continue
            
            # Get contract context (may be truncated in CUAD)
            context = item.get('context', '')
            if not context:
                continue
            
            # Get answers
            answers = item.get('answers', {})
            answer_texts = answers.get('text', [])
            
            # Get first answer or "NOT FOUND"
            if answer_texts and answer_texts[0]:
                clause_text = answer_texts[0]
            else:
                clause_text = "NOT FOUND"
            
            # Truncate context if too long (keep first 4000 chars for LLM context limits)
            if len(context) > 4000:
                context = context[:4000] + "..."
            
            example = dspy.Example(
                contract_text=context,
                clause_type=clause_type,
                clause_text=clause_text
            ).with_inputs('contract_text', 'clause_type')
            
            examples.append(example)
        
        return examples
    
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
        print(f"Successfully loaded CUAD dataset.")
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
