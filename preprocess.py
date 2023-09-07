import numpy as np 
import cv2 
import os 
from sklearn.model_selection import train_test_split

class_names = {'bajra': 0, 'castor': 1, 'cotton': 2, 'paddy': 3, 'sugarcane': 4, 'wheat': 5}

## return numpy array from images and relevant labels
def read_images_and_labels(data_dir): 
    data, labels = [], []
    for class_name in sorted(os.listdir(data_dir)): 
        label = class_names[class_name]
        for image in os.listdir(os.path.join(data_dir, class_name)): 
            image_path = os.path.join(data_dir, class_name, image)
            img = cv2.imread(image_path)
            if img is not None: 
                img = cv2.resize(img, (256,256))
                img = img.astype(np.float32) / 255.0 
                data.append(img)
                labels.append(label)
    return np.array(data), np.array(labels)


