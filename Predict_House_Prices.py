# This program was created by Joseph Lee.
# Documentation updated by Kevin Dzitkowski.
# GitHub Repository: https://github.com/josephlee94/intuitive-deep-learning

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example data
X_train = [[1200], [1300], [1400], [1500], [1600]]
y_train = [[300000], [320000], [340000], [360000], [380000]]

# Model definition
model = Sequential([
    Dense(10, input_dim=1, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=1)

# Make predictions
test_data = [[1700]]
predicted_price = model.predict(test_data)
print(f"Predicted price for 1700 sqft house: ${predicted_price[0][0]:.2f}")
