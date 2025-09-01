# This program was created by Joseph Lee. 
# Documentation updated by Kevin Dzitkowski. 
# GitHub Repository: https://github.com/josephlee94/intuitive-deep-learning


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
y = y / np.max(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae}")

input("Press Enter to exit...")

