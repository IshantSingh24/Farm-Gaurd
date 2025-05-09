import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

model = tf.keras.models.load_model(r"C:\Users\thund\OneDrive\Desktop\tomato leaf disease\model.keras")

class_names = [
    "Bacterial spot", "Early blight", "Late blight", "Leaf Mold", 
    "Septoria leaf spot", "Target Spot", "Spider mites", 
    "Yellow Leaf Curl Virus", "Tomato mosaic virus", "Healthy"
]

cure_methods = {
    "Bacterial spot": [
        "🛑 Remove infected leaves immediately.",
        "🚿 Avoid overhead watering to prevent bacterial spread.",
        "🧼 Disinfect gardening tools regularly.",
        "🛡 Apply **copper-based fungicides** to protect healthy plants."
    ],
    "Early blight": [
        "🌱 Rotate crops every season to prevent disease buildup.",
        "🪓 Remove affected leaves to stop further spread.",
        "💦 Water plants at the base (not leaves) to keep foliage dry.",
        "🛡 Use **fungicides like chlorothalonil** for protection."
    ],
    "Healthy": [
        "✅ Your plant is healthy! Keep up the good care. 🌿",
        "💡 Regularly check for pests and diseases.",
        "💧 Provide proper water and sunlight for continued growth."
    ],
    "Late blight": [
        "🛑 Remove and destroy infected plants to stop spreading.",
        "💦 Avoid watering leaves—keep plants dry.",
        "🛡 Apply **fungicides like mancozeb or copper-based sprays.**"
    ],
    "Leaf Mold": [
        "💨 Increase airflow by properly spacing plants.",
        "🔆 Provide **good sunlight and ventilation.**",
        "🛡 Apply **fungicides like copper spray** if needed."
    ],
    "Septoria leaf spot": [
        "🌿 Remove affected leaves to slow the disease.",
        "🚿 Water at the base to keep leaves dry.",
        "🛡 Use **fungicides like chlorothalonil or copper-based sprays.**"
    ],
    "Spider mites": [
        "🪴 Spray plants with **neem oil or insecticidal soap.**",
        "💦 Keep plants well-watered (mites prefer dry conditions).",
        "🛑 Remove heavily infested leaves."
    ],
    "Target Spot": [
        "🛑 Remove infected leaves to stop the spread.",
        "🛡 Apply **fungicides** recommended for fungal diseases.",
        "💨 Improve airflow around plants."
    ],
    "Tomato mosaic virus": [
        "🚫 Remove infected plants immediately (no cure).",
        "🧼 Disinfect gardening tools after every use.",
        "🐞 Control aphids and insects that spread the virus."

        
    ],
    "Yellow Leaf Curl Virus": [
        "🦟 Control **whiteflies** using sticky traps or neem oil.",
        "🚫 Remove infected plants to protect healthy ones.",
        "🌱 Plant **resistant varieties** if available."
    ]
}

def predict_image(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_label = class_names[predicted_class]
    return class_label

st.title("🍅 Tomato Leaf Disease Detector")

uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_img_path = "temp.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(temp_img_path, caption="📸 Uploaded Image", use_column_width=True)
    class_label = predict_image(temp_img_path, model, class_names)
    st.write(f"### 🏷 **Diagnosis:** **{class_label}**")
    st.write("### 💊 **Recommended Treatment:**")
    for step in cure_methods[class_label]:
        st.write(f"- {step}")
    os.remove(temp_img_path)
