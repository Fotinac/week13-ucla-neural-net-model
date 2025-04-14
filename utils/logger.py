
import logging
import os

def get_logger(name: str) -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
