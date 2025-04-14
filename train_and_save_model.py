
from utils.logger import get_logger
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from models.neural_net_model import build_model
import pandas as pd

logger = get_logger(__name__)

def train_and_save_model():
    try:
        df = load_data("data/Admission.csv")
        df_clean = preprocess_data(df)

        X = df_clean.drop("Chance of Admit ", axis=1)
        y = df_clean["Chance of Admit "]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = build_model(input_shape=X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)

        model.save("models/admission_model.h5")
        logger.info("Model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Failed to train and save model: {e}")
        raise e
