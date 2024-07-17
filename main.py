import tensorflow as tf
from tensorflow import keras

# Load the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#Normalize the data
X_train = X_train/255
X_test = X_test/255

# Create the model
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')  # Output layer with 10 neurons for 10 classes
    ]
)

model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(X_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(X_test, y_test)

# Save model
model.save("digit_recognition_model.h5")