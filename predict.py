## Selecting random images from directory and make prediction


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


class_names = {'bajra': 0, 'castor': 1, 'cotton': 2, 'paddy': 3, 'sugarcane': 4, 'wheat': 5}

## loading saved model 
model = load_model(r'C:\Users\abc\Desktop\agri_bajra_castor_cotton\models\model_optimization_resnet50.h5')

## image directory
test_images_dir = r'C:\Users\abc\Desktop\agri_bajra_castor_cotton\dataset\castor'  
image_file_names = os.listdir(test_images_dir)

## picking up images and make prediction
num_images_to_show = 10
selected_images = np.random.choice(image_file_names, num_images_to_show, replace=True)

for image_file in selected_images:
    image_path = os.path.join(test_images_dir, image_file)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  
    predictions = model.predict(img)
    predicted_label = class_names[np.argmax(predictions)]
    predicted_probability = predictions.max()

 ## Plotting
    plt.figure()
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f'Original Label: Unknown\nPredicted Label: {predicted_label}\nPredicted Probability: {predicted_probability:.2f}')
    plt.axis('off')
    plt.show()
