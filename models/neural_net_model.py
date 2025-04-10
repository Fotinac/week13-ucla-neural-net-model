import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    """
    Build a simple feedforward neural network with 2 hidden layers.

    Args:
        input_shape (int): Number of input features

    Returns:
        model (Sequential): Compiled Keras model
    """
    try:
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        logging.info("Neural network model built and compiled successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building model: {e}")
        raise

def train_model(model, X_train, y_train, epochs=50, batch_size=32, verbose=1):
    """
    Train the neural network model.

    Returns:
        history: Training history object
    """
    try:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        logging.info("Model training completed.")
        return model, history
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
