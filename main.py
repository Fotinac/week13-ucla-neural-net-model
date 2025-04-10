import logging
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from models.neural_net_model import build_model, train_model
from utils.evaluator import evaluate_model
from utils.visualizer import plot_training_history
import matplotlib.pyplot as plt

# === Setup Logging ===
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("Starting pipeline...")

    try:
        # 1. Load data
        data_path = "data/Admission.csv"
        df = load_data(data_path)

        # 2. Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df, target_column="Admit_Chance")

        # 3. Build model
        model = build_model(input_shape=X_train.shape[1])

        # 4. Train model
        model, history = train_model(model, X_train, y_train, epochs=50)

        # 5. Evaluate
        results = evaluate_model(model, X_test, y_test)
        print(f"\n Accuracy: {results['accuracy']:.2f}")
        print("\n Classification Report:\n", results["classification_report"])
        print("Confusion Matrix:\n", results["confusion_matrix"])

        # 6. Visualize training history
        fig = plot_training_history(history)
        plt.show()

    except Exception as e:
        logging.error(f"Unhandled error in main(): {e}")
        print("An error occurred. Check logs/app.log for details.")

# Ensure script runs when executed directly
if __name__ == "__main__":
    main()
