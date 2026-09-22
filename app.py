
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

MODEL_PATH = "fashion_mnist_cnn.keras"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fashion-MNIST CNN Classifier",
    page_icon="👕",
    layout="centered"
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("👕 Fashion-MNIST CNN Image Classifier")

st.write(
    """
    Upload an image of a clothing item and the trained CNN model
    will classify it into one of 10 Fashion-MNIST categories.
    """
)

st.info(
    "The model was trained on 28×28 grayscale Fashion-MNIST images. "
    "For best results, upload a photo containing one clearly visible item."
)


# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded image")
    st.image(
        image,
        caption="Image selected for classification",
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    st.subheader("Image preprocessing")

    processed_image = image.convert("L")
    processed_image = processed_image.resize((28, 28))
    
    # Invert the image to match the Fashion-MNIST format
    processed_image = Image.eval(processed_image, lambda pixel: 255 - pixel)
    
    image_array = np.array(processed_image, dtype=np.float32)
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)

    st.write(
        "The image was converted to grayscale, resized to 28×28 pixels, "
        "normalized to the range 0–1, and prepared for the CNN."
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    st.subheader("Classification")

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted class",
            predicted_class
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

    # -----------------------------------------------------
    # Probability visualization
    # -----------------------------------------------------

    st.subheader("Class probabilities")

    probability_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": predictions
    })

    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)

    st.bar_chart(
        probability_df.set_index("Class")
    )

    st.caption(
        "The confidence represents the probability assigned by the CNN "
        "to the predicted class."
    )
