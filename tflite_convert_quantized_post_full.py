import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random

# Create datasets and load data by batch
class Dataset:
    def __init__(self, data, label):
        self.data = data # the paths of images
        self.label = label # the paths of segmentation images
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = size
        
    def __getitem__(self, index):
        # collect batch data
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        data = []
        for i in range(start, stop):
            data.append(self.dataset[i])
            
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)
    
    def __len__(self):
        return self.size // self.batch_size 

# Load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train/255
X_test = X_test/255

train_dataset = Dataset(X_train, y_train)
train_loader = DataLoader(train_dataset, 1, len(train_dataset))

#
def representative_data_gen():
    for idx in range(len(train_loader)):
        data = train_loader.__getitem__(idx)
        yield [np.array(data[0], dtype=np.float32, ndmin=2)]

# load model 
digit_recognition_model = tf.keras.models.load_model("digit_recognition_model.h5") 
print("Size model digit_recognition_model: ", digit_recognition_model.count_params()*4)

# Create converter for convert model
try: 
    # Create converter for convert model
    converter = tf.lite.TFLiteConverter.from_keras_model(digit_recognition_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # OPTIMIZE_FOR_SIZE, OPTIMIZE_FOR_LATENCY
    converter.representative_dataset = representative_data_gen
    tflite_model_quantized_full = converter.convert()
    print("Size model converted: ", len(tflite_model_quantized_full))

    with open("tflite_model_quantized_full.tflite", "wb") as f:
        f.write(tflite_model_quantized_full)
        
    print("Model converted successfully.")
    
except OSError as e:
    print(f"Error: {e}")
    print("Please ensure the path to the SavedModel is correct and the directory contains the saved_model.pb or saved_model.pbtxt file.")

