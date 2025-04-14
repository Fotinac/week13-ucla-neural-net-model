
from utils.logger import get_logger
from train_and_save_model import train_and_save_model

logger = get_logger(__name__)

def main():
    try:
        logger.info("Application started.")
        train_and_save_model()
        logger.info("Application finished successfully.")
    except Exception as e:
        logger.error(f"Application failed with error: {e}")

if __name__ == "__main__":
    main()
