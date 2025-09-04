import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random

# Function to apply low-light and quality degradation
def degrade_image(image):
    # Convert to low-light (reduce brightness)
    low_light = image * random.uniform(0.3, 0.7)  # Reduce brightness by 30-70%
    low_light = np.clip(low_light, 0, 1)
    
    # Adds Gaussian blur to simulate poor focus
    if random.random() > 0.5:  # 50% chance to apply blur
        kernel_size = random.choice([3, 5])
        low_light = cv2.GaussianBlur(low_light, (kernel_size, kernel_size), 0)
    
    # Adds noise
    if random.random() > 0.3:  # 70% chance to add noise
        noise = np.random.normal(0, 0.05, low_light.shape)
        low_light = low_light + noise
        low_light = np.clip(low_light, 0, 1)
    
    # Randomly adjust contrast
    if random.random() > 0.4:  # 60% chance to adjust contrast
        low_light = low_light * random.uniform(0.7, 1.3)
        low_light = np.clip(low_light, 0, 1)
    
    return low_light

# Custom data generator that applies degradation
def degraded_generator(generator):
    while True:
        batch_x, batch_y = next(generator)
        degraded_batch = np.array([degrade_image(img) for img in batch_x])
        yield degraded_batch, batch_y

# Loads trained model
model = tf.keras.models.load_model(r"C:\Users\thund\OneDrive\Desktop\Farm Gaurd\model.keras")

class_names = [
    "Bacterial spot", "Early blight", "Late blight", "Leaf Mold", 
    "Septoria leaf spot", "Target Spot", "Spider mites", 
    "Yellow Leaf Curl Virus", "Tomato mosaic virus", "Healthy"
]

# Path to your test dataset
test_dir = r"C:\Users\thund\OneDrive\Desktop\Farm Gaurd\archive\tomato\val"  

# Image parameters
img_size = (128, 128)
batch_size = 32

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Create degraded version of test data
degraded_test_gen = degraded_generator(test_generator)

# Get true labels (same as original)
y_true = test_generator.classes

# Get predictions on degraded images
# Need to reset generator first
test_generator.reset()
steps = len(test_generator)
y_pred = model.predict(degraded_test_gen, steps=steps)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix for Tomato Leaf Disease Classification (Low-Light Conditions)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('confusion_matrix_low_light.png')

accuracy = np.trace(cm) / np.sum(cm)
print(f"Overall Accuracy under low-light conditions: {accuracy:.2%}")

test_generator.reset()
y_pred_original = model.predict(test_generator)
y_pred_original_classes = np.argmax(y_pred_original, axis=1)
original_accuracy = np.sum(y_true == y_pred_original_classes) / len(y_true)
print(f"Original Accuracy (normal conditions): {original_accuracy:.2%}")
print(f"Accuracy drop due to low-light conditions: {(original_accuracy - accuracy):.2%}")