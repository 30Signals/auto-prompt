import pandas as pd
import dspy
import os
from . import config

def load_data():
    """
    Loads resume data from CSV and returns train and test splits as dspy.Example objects.
    
    Split logic:
    - Train: First 20 samples
    - Test: Next 30 samples
    - Rest: Ignored
    """
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {config.DATA_PATH}")

    df = pd.read_csv(config.DATA_PATH)
    
    # Ensure required columns exist
    required_columns = ['resume_id', 'unstructured_text', 'job_role', 'skills', 'education', 'experience_years']
    for col in required_columns:
        if col not in df.columns:
             raise ValueError(f"Missing required column: {col}")

    # Deterministic Split
    train_df = df.iloc[:config.TRAIN_SIZE]
    test_df = df.iloc[config.TRAIN_SIZE : config.TRAIN_SIZE + config.TEST_SIZE]

    # Convert to dspy.Example
    # Input: unstructured_text
    # Labels: job_role, skills, education, experience_years
    
    # Convert to dspy.Example using list comprehension for better performance
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
    # Test loading
    try:
        train, test = load_data()
        print(f"Successfully loaded data.")
        print(f"Train size: {len(train)}")
        print(f"Test size: {len(test)}")
        print("Sample Train Item:")
        print(train[0])
    except Exception as e:
        print(f"Error loading data: {e}")
