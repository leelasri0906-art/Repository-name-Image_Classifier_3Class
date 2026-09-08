import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🤖",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: #f5f5f5;
    margin-top: 20px;
}

.prediction {
    font-size: 36px;
    font-weight: bold;
}

.confidence {
    font-size: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.markdown(
    '<div class="title">🤖 AI Image Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image and AI will classify it as a Car, Cat, or Dog.'
    '</div>',
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "model/classifier.keras"
    )

    return model


@st.cache_data
def load_classes():

    with open("model/classes.txt", "r") as file:
        classes = [line.strip() for line in file.readlines()]

    return classes


model = load_model()
class_names = load_classes()

# =========================
# CLASS INFORMATION
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🚗 Car")

with col2:
    st.info("🐱 Cat")

with col3:
    st.info("🐶 Dog")

# =========================
# IMAGE UPLOADER
# =========================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    # Resize
    image_resized = image.resize((180, 180))

    # Convert to array
    image_array = np.array(image_resized)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    with st.spinner("AI is analyzing the image..."):

        predictions = model.predict(
            image_array,
            verbose=0
        )

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    ) * 100

    # =========================
    # RESULT
    # =========================

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.write("### Prediction")

    st.markdown(
        f'<div class="prediction">'
        f'{predicted_class.title()}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence">'
        f'Confidence: {confidence:.2f}%'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # =========================
    # ALL PREDICTIONS
    # =========================

    st.write("### 📊 Prediction Probabilities")

    probabilities = predictions[0] * 100

    for class_name, probability in zip(
        class_names,
        probabilities
    ):

        st.write(
            f"**{class_name.title()}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            float(probability / 100)
        )