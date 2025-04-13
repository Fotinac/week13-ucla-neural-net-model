import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

# Load dataset
df = pd.read_csv("data/Admission.csv")
df = df.drop(columns=["Serial_No"])  # Drop ID column if present

X = df.drop(columns=["Admit_Chance"])
y = df["Admit_Chance"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # For probabilities

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/admission_model.h5")
print("Model trained and saved to models/admission_model.h5")
