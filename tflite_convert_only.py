import tensorflow as tf

# load model 
digit_recognition_model = tf.keras.models.load_model("digit_recognition_model.h5") 
print("Size model digit_recognition_model: ", digit_recognition_model.count_params()*4)

# Create converter for convert model
converter = tf.lite.TFLiteConverter.from_keras_model(digit_recognition_model)
tflite_model = converter.convert()
print("Size model converted: ", len(tflite_model))

with open("tflite_model.tflite", "wb") as f:
    f.write(tflite_model)
    
