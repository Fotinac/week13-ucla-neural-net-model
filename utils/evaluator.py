import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print performance metrics.
    
    Returns:
        dict: accuracy, classification_report, confusion_matrix
    """
    try:
        predictions = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)

        logging.info(f"Evaluation complete. Accuracy: {acc:.4f}")
        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": matrix
        }
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise
