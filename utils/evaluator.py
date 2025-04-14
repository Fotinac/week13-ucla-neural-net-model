
from sklearn.metrics import accuracy_score
from utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        logger.info(f"Model accuracy: {acc}")
        return acc
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise e
