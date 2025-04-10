import pandas as pd
import logging

def load_data(filepath):
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {filepath}: {e}")
        raise
