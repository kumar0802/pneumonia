import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

MODEL_PATH = Path("xray.keras")  # Change this if your model has another name
CLASS_NAMES_PATH = Path("classes.json")
IMAGE_SIZE = (64, 64)           # Must match the size used during training


# ---------------------------------------------------------
# Load model and class names only once
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place CAT.keras in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}. "
            "Create classes.json during training."
        )
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list) or not names:
        raise ValueError("classes.json must contain a non-empty list of class names.")
    return names


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_image(image: Image.Image, model, class_names):
    image = image.convert("RGB").resize(IMAGE_SIZE)

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)
    probabilities = np.asarray(predictions)[0]

    if probabilities.ndim != 1:
        raise ValueError("The model returned an unexpected prediction shape.")

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model has {len(probabilities)} output classes, but "
            f"classes.json contains {len(class_names)} names. "
            "Make sure both came from the same training run."
        )

    predicted_index = int(np.argmax(probabilities))
    return class_names[predicted_index], float(probabilities[predicted_index]), probabilities


# ---------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------
st.title("🖼️ CNN Image Classification")
st.write("Upload an image to classify it using your trained TensorFlow CNN model.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

try:
    model = load_model()
    class_names = load_class_names()

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, caption="Image selected for prediction", use_container_width=True)

        if st.button("Predict Image", type="primary", use_container_width=True):
            with st.spinner("Classifying image..."):
                predicted_class, confidence, probabilities = predict_image(
                    image, model, class_names
                )

            st.success(f"Predicted class: {predicted_class}")
            st.metric("Model confidence", f"{confidence * 100:.2f}%")

            st.subheader("Class probabilities")
            probability_data = {
                class_names[i]: float(probabilities[i])
                for i in range(len(class_names))
            }
            st.bar_chart(probability_data)

    else:
        st.info("Upload an image to get started.")

except (FileNotFoundError, ValueError, OSError) as error:
    st.error(str(error))
except Exception as error:
    st.error(f"Something went wrong while loading or predicting: {error}")

st.caption("CNN image classifier | TensorFlow + Streamlit")
