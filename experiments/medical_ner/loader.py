"""
Data Loading for Medical NER Experiment

Downloads and parses the NCBI Disease Corpus from Hugging Face.
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


def download_ncbi_corpus():
    """
    Download NCBI Disease Corpus from Hugging Face.
    Uses local cache to avoid repeated downloads.
    
    Returns:
        Dataset with train, validation, and test splits
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
        "ncbi/ncbi_disease",
        revision="refs/convert/parquet",
        cache_dir=str(CACHE_DIR),
    )
    return dataset


def parse_ner_tags(tokens, ner_tags):
    """
    Parse BIO-tagged tokens into disease entity list.
    
    Args:
        tokens: List of tokens
        ner_tags: List of NER tags (0=O, 1=B-Disease, 2=I-Disease)
    
    Returns:
        List of disease entity strings
    """
    diseases = []
    current_disease = []
    
    for token, tag in zip(tokens, ner_tags):
        if tag == 1:  # B-Disease (beginning of disease)
            if current_disease:
                diseases.append(" ".join(current_disease))
            current_disease = [token]
        elif tag == 2:  # I-Disease (inside disease)
            current_disease.append(token)
        else:  # O (outside)
            if current_disease:
                diseases.append(" ".join(current_disease))
                current_disease = []
    
    # Don't forget last entity
    if current_disease:
        diseases.append(" ".join(current_disease))
    
    return diseases


def load_data(train_size=None, test_size=None, seed=None):
    """
    Load NCBI Disease Corpus and return train/test splits as dspy.Example objects.
    
    Args:
        train_size: Number of training samples. Default: config.TRAIN_SIZE
        test_size: Number of test samples. Default: config.TEST_SIZE
        seed: Random seed for shuffling. Default: None
    
    Returns:
        Tuple of (trainset, testset) as lists of dspy.Example objects
    """
    train_size = train_size or config.TRAIN_SIZE
    test_size = test_size or config.TEST_SIZE
    
    # Download dataset
    dataset = download_ncbi_corpus()
    
    # Get train and test splits
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Shuffle if seed provided
    if seed is not None:
        train_data = train_data.shuffle(seed=seed)
        test_data = test_data.shuffle(seed=seed)
    
    # Select subset
    train_data = train_data.select(range(min(train_size, len(train_data))))
    test_data = test_data.select(range(min(test_size, len(test_data))))
    
    def convert_to_examples(data):
        examples = []
        for item in data:
            tokens = item['tokens']
            ner_tags = item['ner_tags']
            
            # Reconstruct abstract text
            abstract_text = " ".join(tokens)
            
            # Parse disease entities
            diseases = parse_ner_tags(tokens, ner_tags)
            diseases_str = ", ".join(diseases) if diseases else ""
            
            example = dspy.Example(
                abstract_text=abstract_text,
                diseases=diseases_str
            ).with_inputs('abstract_text')
            
            examples.append(example)
        
        return examples
    
    trainset = convert_to_examples(train_data)
    testset = convert_to_examples(test_data)
    
    return trainset, testset


if __name__ == "__main__":
    try:
        train, test = load_data()
        print(f"Successfully loaded NCBI Disease Corpus.")
        print(f"Train size: {len(train)}")
        print(f"Test size: {len(test)}")
        print("\nSample Train Item:")
        print(f"Abstract: {train[0].abstract_text[:200]}...")
        print(f"Diseases: {train[0].diseases}")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
