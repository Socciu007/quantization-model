import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

# Load the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#Normalize the data
X_train = X_train/255
X_test = X_test/255

# load model 
digit_recognition_model = tf.keras.models.load_model("digit_recognition_model.h5") 
digit_recognition_model.summary()

# Create model quantized
q_digit_recognition_model = tfmot.quantization.keras.quantize_model(digit_recognition_model)
q_digit_recognition_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

q_digit_recognition_model.summary()

q_digit_recognition_model.fit(X_train, y_train, epochs = 1)

# Evaluate the model quantized
q_digit_recognition_model.evaluate(X_test, y_test)

# Convert tflite
converter = tf.lite.TFLiteConverter.from_keras_model(q_digit_recognition_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_q_aware_model = converter.convert()

print(len(tflite_q_aware_model))
with open("tflite_q_aware_model.tflite", 'wb') as f:
    f.write(tflite_q_aware_model)