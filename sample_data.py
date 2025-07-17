"""
Sample dataset generator for testing the outlier detection system
Creates synthetic datasets with known outliers for demonstration purposes
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
import matplotlib.pyplot as plt

def generate_sample_dataset(n_samples=1000, n_features=4, n_outliers=50, random_state=42):
    """
    Generate a sample dataset with known outliers for testing
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_outliers: Number of outliers to inject
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with generated data
    """
    np.random.seed(random_state)
    
    # Generate normal data using make_blobs
    normal_data, _ = make_blobs(
        n_samples=n_samples - n_outliers,
        centers=3,
        n_features=n_features,
        cluster_std=1.0,
        random_state=random_state
    )
    
    # Generate outliers (points far from normal clusters)
    outlier_data = np.random.uniform(
        low=normal_data.min() - 3,
        high=normal_data.max() + 3,
        size=(n_outliers, n_features)
    )
    
    # Combine normal and outlier data
    all_data = np.vstack([normal_data, outlier_data])
    
    # Create labels (0 for normal, 1 for outlier)
    labels = np.hstack([
        np.zeros(n_samples - n_outliers),
        np.ones(n_outliers)
    ])
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(all_data, columns=feature_names)
    df['is_outlier'] = labels
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df

def generate_iris_like_dataset(n_samples=150, random_state=42):
    """
    Generate an Iris-like dataset for testing
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
    
    Returns:
        DataFrame with Iris-like data
    """
    np.random.seed(random_state)
    
    # Generate base features
    sepal_length = np.random.normal(5.8, 0.8, n_samples)
    sepal_width = np.random.normal(3.1, 0.4, n_samples)
    petal_length = np.random.normal(3.8, 1.8, n_samples)
    petal_width = np.random.normal(1.2, 0.8, n_samples)
    
    # Add some outliers
    n_outliers = int(n_samples * 0.1)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    
    # Make some features extreme for outliers
    sepal_length[outlier_indices] += np.random.normal(0, 2, n_outliers)
    sepal_width[outlier_indices] += np.random.normal(0, 1, n_outliers)
    petal_length[outlier_indices] += np.random.normal(0, 2, n_outliers)
    petal_width[outlier_indices] += np.random.normal(0, 1, n_outliers)
    
    # Create DataFrame
    df = pd.DataFrame({
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    })
    
    return df

def generate_financial_dataset(n_samples=500, random_state=42):
    """
    Generate a financial dataset for fraud detection simulation
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
    
    Returns:
        DataFrame with financial data
    """
    np.random.seed(random_state)
    
    # Normal transactions
    normal_amount = np.random.lognormal(mean=3, sigma=1, size=int(n_samples * 0.95))
    normal_frequency = np.random.poisson(lam=5, size=int(n_samples * 0.95))
    normal_time = np.random.uniform(8, 20, size=int(n_samples * 0.95))  # Business hours
    
    # Fraudulent transactions (outliers)
    n_fraud = int(n_samples * 0.05)
    fraud_amount = np.random.lognormal(mean=6, sigma=1, size=n_fraud)  # Higher amounts
    fraud_frequency = np.random.poisson(lam=15, size=n_fraud)  # More frequent
    fraud_time = np.random.uniform(0, 24, size=n_fraud)  # Any time
    
    # Combine data
    amount = np.hstack([normal_amount, fraud_amount])
    frequency = np.hstack([normal_frequency, fraud_frequency])
    time_of_day = np.hstack([normal_time, fraud_time])
    
    # Additional features
    account_age = np.random.exponential(scale=365, size=n_samples)
    location_risk = np.random.beta(a=1, b=4, size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_amount': amount,
        'daily_frequency': frequency,
        'time_of_day': time_of_day,
        'account_age_days': account_age,
        'location_risk_score': location_risk
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate sample datasets
    
    # Basic outlier dataset
    basic_data = generate_sample_dataset()
    basic_data.to_csv('basic_outlier_dataset.csv', index=False)
    print("Basic outlier dataset created: basic_outlier_dataset.csv")
    
    # Iris-like dataset
    iris_data = generate_iris_like_dataset()
    iris_data.to_csv('iris_like_dataset.csv', index=False)
    print("Iris-like dataset created: iris_like_dataset.csv")
    
    # Financial dataset
    financial_data = generate_financial_dataset()
    financial_data.to_csv('financial_dataset.csv', index=False)
    print("Financial dataset created: financial_dataset.csv")
    
    print("\nDataset Statistics:")
    print(f"Basic dataset: {basic_data.shape[0]} rows, {basic_data.shape[1]} columns")
    print(f"Iris dataset: {iris_data.shape[0]} rows, {iris_data.shape[1]} columns")  
    print(f"Financial dataset: {financial_data.shape[0]} rows, {financial_data.shape[1]} columns")