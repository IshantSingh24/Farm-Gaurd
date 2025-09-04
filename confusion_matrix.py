import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\thund\OneDrive\Desktop\Farm Gaurd\model.keras")

# Define your class names (same as in your Streamlit app)
class_names = [
    "Bacterial spot", "Early blight", "Late blight", "Leaf Mold", 
    "Septoria leaf spot", "Target Spot", "Spider mites", 
    "Yellow Leaf Curl Virus", "Tomato mosaic virus", "Healthy"
]

# Path to your test dataset (should have subfolders for each class)
test_dir = r"C:\Users\thund\OneDrive\Desktop\Farm Gaurd\archive\tomato\val"  

# Image parameters (should match your training parameters)
img_size = (128, 128)
batch_size = 32

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for confusion matrix
)

# Get true labels and predictions
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix for Tomato Leaf Disease Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the figure
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Calculate and print accuracy
accuracy = np.trace(cm) / np.sum(cm)
print(f"Overall Accuracy: {accuracy:.2%}")