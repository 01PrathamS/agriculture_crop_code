import os 
import cv2 
import numpy as np 
from preprocess import read_images_and_labels
from sklearn.model_selection import train_test_split 
import tensorflow as tf  
from tensorflow.keras import layers, models, optimizers 
from models import resnet1
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt 


## 
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
saved_model_path = os.path.join(root_dir, 'models', 'cotton_castor_binary_resnet1_deepak.h5')

## All About data Preparation

dataset_dir = r'C:\Users\abc\Desktop\agri_bajra_castor_cotton\dataset'

train_data, train_labels = read_images_and_labels(dataset_dir)

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, 
                                                                    test_size=0.2, 
                                                                    random_state=42)

print(len(train_data), len(test_data), len(train_labels), len(test_labels))

# Convert labels to one hot encoding 
num_classes = len(np.unique(train_labels))
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)


### All about Modeling

input_shape = (256, 256, 3)
num_classes = 2
n_epochs=10
lr = 0.0005

def scheduler(epoch, lr):
    if epoch == 0:
        return lr
    return 0.5 * lr * (1 + np.cos(epoch * np.pi / n_epochs))

checkpoint = ModelCheckpoint(saved_model_path, monitor='loss', verbose=1, save_best_only=False, mode='min')
learning_rate_scheduler = LearningRateScheduler(scheduler, verbose=1)
callbacks_list = [checkpoint, learning_rate_scheduler]

model = resnet1(input_shape, num_classes)

model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=n_epochs,
                     validation_data=(test_data, test_labels), 
                     callbacks=callbacks_list)

test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

model.save(saved_model_path)

