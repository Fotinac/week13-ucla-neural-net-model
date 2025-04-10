import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column="Admit_Chance", test_size=0.2, random_state=42):
    """
    Preprocess the UCLA admission dataset.

    - Converts target to binary classification (e.g., admit or not)
    - Normalizes feature columns
    - Splits into train/test

    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        df = df.copy()
        
        # Binary classification: 1 if chance >= 0.75, else 0
        df[target_column] = (df[target_column] >= 0.75).astype(int)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

        logging.info("Data preprocessed and split successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise
