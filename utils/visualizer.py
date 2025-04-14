
import matplotlib.pyplot as plt
from utils.logger import get_logger

logger = get_logger(__name__)

def plot_feature_importance(importances, feature_names, output_path):
    try:
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances)
        plt.xlabel("Importance")
        plt.title("Feature Importances")
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")
        raise e
