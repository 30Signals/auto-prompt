"""
Data Loading for Resume Extraction Experiment
"""

import pandas as pd
import dspy
from pathlib import Path
from . import config


def load_data(data_path=None, train_size=None, test_size=None):
    """
    Load resume data from CSV and return train/test splits as dspy.Example objects.

    Args:
        data_path: Path to CSV file. Default: config.DEFAULT_DATA_PATH
        train_size: Number of training samples. Default: config.TRAIN_SIZE
        test_size: Number of test samples. Default: config.TEST_SIZE

    Returns:
        Tuple of (trainset, testset) as lists of dspy.Example objects
    """
    data_path = Path(data_path or config.DEFAULT_DATA_PATH)
    train_size = train_size or config.TRAIN_SIZE
    test_size = test_size or config.TEST_SIZE

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)

    # Validate required columns
    required_columns = ['resume_id', 'unstructured_text', 'job_role', 'skills', 'education', 'experience_years']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Deterministic split
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:train_size + test_size]

    # Convert to dspy.Example
    trainset = [
        dspy.Example(
            unstructured_text=row['unstructured_text'],
            job_role=row['job_role'],
            skills=row['skills'],
            education=row['education'],
            experience_years=str(row['experience_years'])
        ).with_inputs('unstructured_text')
        for row in train_df.to_dict('records')
    ]

    testset = [
        dspy.Example(
            unstructured_text=row['unstructured_text'],
            job_role=row['job_role'],
            skills=row['skills'],
            education=row['education'],
            experience_years=str(row['experience_years'])
        ).with_inputs('unstructured_text')
        for row in test_df.to_dict('records')
    ]

    return trainset, testset


if __name__ == "__main__":
    try:
        train, test = load_data()
        print(f"Successfully loaded data.")
        print(f"Train size: {len(train)}")
        print(f"Test size: {len(test)}")
        print("\nSample Train Item:")
        print(train[0])
    except Exception as e:
        print(f"Error loading data: {e}")
