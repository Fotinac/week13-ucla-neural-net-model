
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_cleaned = df.dropna()
        logger.info(f"Preprocessed data. Original shape: {df.shape}, New shape: {df_cleaned.shape}")
        return df_cleaned
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise e
