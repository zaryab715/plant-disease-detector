import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f4faf5 0%,
        #eaf5ec 100%
    );
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Header */

.header-card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid #dceadd;
    margin-bottom: 30px;
}

.main-title {
    text-align: center;
    color: #1b5e20;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #55705a;
    font-size: 18px;
}

/* Section titles */

.section-title {
    color: #1b5e20;
    font-size: 25px;
    font-weight: 650;
    margin-bottom: 15px;
}

/* Cards */

.info-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #dceadd;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.06);
}

/* Upload box */

[data-testid="stFileUploader"] {
    background: white;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #dceadd;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
}

/* Image */

[data-testid="stImage"] img {
    border-radius: 15px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #edf7ef;
    border-right: 1px solid #d3e5d6;
}

/* Buttons */

.stButton > button {
    border-radius: 10px;
}

/* Footer */

.custom-footer {
    text-align: center;
    color: #6b7d6d;
    font-size: 14px;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid #d8e5da;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = "models/early_blight_weighted.keras"

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_healthy"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:

    model = load_model()
    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error("Unable to load the trained model.")
    st.error(str(e))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌿 Plant Disease Detector")

    st.markdown("---")

    st.markdown("### About the Model")

    st.write(
        "This application uses a trained MobileNetV2 "
        "deep learning model to classify tomato leaf images."
    )

    st.markdown("### Model Information")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Number of Classes:** 6")
    st.write("**Test Accuracy:** 94.53%")
    st.write("**Early Blight Recall:** 80.00%")
    st.write("**Macro Recall:** 93.80%")
    st.write("**Macro F1:** 93.76%")

    st.markdown("---")

    st.markdown("### Supported Diseases")

    st.write("🍃 Bacterial Spot")
    st.write("🍂 Early Blight")
    st.write("🍁 Late Blight")
    st.write("🌱 Leaf Mold")
    st.write("🍃 Septoria Leaf Spot")
    st.write("🌿 Healthy")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="header-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🌿 Plant Disease Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered tomato leaf disease classification using MobileNetV2</div>',
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    st.markdown("### 📷 Upload a Tomato Leaf Image")

    st.write(
        "Upload a clear image of a tomato leaf and "
        "the trained model will predict its most likely class."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG image."
    )


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None and model_loaded:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown("---")

    image_col, result_col = st.columns([1, 1])


    # ========================================================
    # UPLOADED IMAGE
    # ========================================================

    with image_col:

        st.markdown("### 🖼️ Uploaded Image")

        st.image(
            image,
            caption="Uploaded tomato leaf",
            use_container_width=True
        )


    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    image_resized = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image_resized
    ).astype(
        np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        float(
            predictions[
                predicted_index
            ]
        ) * 100
    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    with result_col:

        st.markdown("### 🔍 Prediction Result")

        st.write(
            f"**Predicted Class:** {predicted_class}"
        )

        st.write(
            f"**Model Confidence:** {confidence:.2f}%"
        )

        st.progress(
            min(
                confidence / 100,
                1.0
            )
        )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.markdown("---")

    st.markdown("### 📊 Class Probabilities")

    for class_name, probability in zip(
        CLASS_NAMES,
        predictions
    ):

        percentage = (
            float(probability) * 100
        )

        col_name, col_value = st.columns(
            [4, 1]
        )

        with col_name:

            st.write(
                class_name
            )

        with col_value:

            st.write(
                f"{percentage:.2f}%"
            )

        st.progress(
            float(probability)
        )


    # ========================================================
    # RESULT INTERPRETATION
    # ========================================================

    st.markdown("---")

    st.markdown("### 💡 Result Interpretation")

    if predicted_class == "Tomato_healthy":

        st.success(
            "The model predicts that the uploaded tomato leaf "
            "belongs to the healthy class."
        )

    else:

        readable_name = (
            predicted_class
            .replace(
                "Tomato_",
                ""
            )
            .replace(
                "_",
                " "
            )
        )

        st.warning(
            f"The model predicts: **{readable_name}**"
        )

        st.info(
            "This prediction is generated by the trained "
            "deep learning model. For important agricultural "
            "decisions, the result should be confirmed by "
            "an appropriate expert."
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    if model_loaded:

        st.markdown("---")

        st.markdown("### 🌱 How to Use")

        st.write(
            "1. Click **Browse files** above."
        )

        st.write(
            "2. Select a tomato leaf image."
        )

        st.write(
            "3. Wait for the model prediction."
        )

        st.write(
            "4. Review the confidence and class probabilities."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌿 Plant Disease Detector • MobileNetV2 • "
    "Tomato Leaf Classification"
)