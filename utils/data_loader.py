
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error for file: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in load_data: {e}")
        raise e
