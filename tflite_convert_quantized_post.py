import tensorflow as tf

# load model 
digit_recognition_model = tf.keras.models.load_model("digit_recognition_model.h5") 
print("Size model digit_recognition_model: ", digit_recognition_model.count_params()*4)

# Create converter for convert model
converter = tf.lite.TFLiteConverter.from_keras_model(digit_recognition_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # OPTIMIZE_FOR_SIZE, OPTIMIZE_FOR_LATENCY
converter.target_spec.supported_types = [tf.float16]
tflite_model_quantized_f16 = converter.convert()
print("Size model converted: ", len(tflite_model_quantized_f16))

with open("tflite_model_quantized_f16.tflite", "wb") as f:
    f.write(tflite_model_quantized_f16)
    
