import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chest X-ray Classification",
    page_icon="🩻",
    layout="centered"
)

MODEL_PATH = Path("xray.keras")
CLASS_NAMES_PATH = Path("classes.json")
IMAGE_SIZE = (64, 64)


# ---------------------------------------------------------
# Load the trained model and class names
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "xray.keras was not found. Put your trained model "
            "in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "classes.json was not found. Put it in the same folder as app.py."
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or len(class_names) == 0:
        raise ValueError("classes.json must contain a non-empty list of class names.")

    return class_names


# ---------------------------------------------------------
# Prepare image and make prediction
# Matches notebook preprocessing:
# load_img(target_size=(64, 64)), img_to_array, expand_dims.
# Rescaling(1./255) is already part of the saved model.
# ---------------------------------------------------------
def predict_xray(image, model, class_names):
    image = image.convert("RGB").resize(IMAGE_SIZE)

    image_array = tf.keras.utils.img_to_array(image)
    image_batch = tf.expand_dims(image_array, axis=0)

    predictions = model.predict(image_batch, verbose=0)
    probabilities = np.asarray(predictions)[0]

    if probabilities.ndim != 1:
        raise ValueError("Unexpected prediction output from the model.")

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"The model returns {len(probabilities)} classes, but "
            f"classes.json contains {len(class_names)} class names."
        )

    predicted_index = int(np.argmax(probabilities))
    return class_names[predicted_index], probabilities


# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------
st.title("🩻 Chest X-ray Classification")
st.write(
    "Upload a chest X-ray image to receive a prediction from your "
    "trained CNN model."
)

st.info(
    "This is an educational machine-learning demonstration, not a medical "
    "diagnosis. A qualified healthcare professional must interpret X-rays."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"]
)

try:
    model = load_model()
    class_names = load_class_names()

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded X-ray")
        st.image(image, caption="Input chest X-ray", use_container_width=True)

        if st.button("Classify X-ray", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                predicted_class, probabilities = predict_xray(
                    image, model, class_names
                )

            confidence = float(np.max(probabilities)) * 100

            st.subheader("Prediction result")
            st.success(f"Predicted class: **{predicted_class}**")
            st.metric("Model confidence", f"{confidence:.2f}%")

            st.subheader("Class-wise prediction scores")
            score_data = {
                class_names[i]: float(probabilities[i]) * 100
                for i in range(len(class_names))
            }
            st.bar_chart(score_data)

            st.caption(
                "Scores are model outputs and should not be interpreted "
                "as clinical probabilities."
            )
    else:
        st.warning("Please upload a chest X-ray image to begin.")

except (FileNotFoundError, ValueError, OSError) as error:
    st.error(str(error))
except Exception as error:
    st.error(f"An unexpected error occurred: {error}")

st.divider()
st.caption("CNN model: TensorFlow / Keras | Input size: 64 × 64")
