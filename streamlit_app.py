import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plant Disease Detection System")
st.divider()

st.markdown(
    """
    Upload a leaf image and the AI model will identify the disease,
    provide confidence scores, and suggest disease information.
    """
)

@st.cache_resource
def load_trained_model():

    model = load_model(
        "saved_models/best_finetuned_model.keras"
    )

    return model

model = load_trained_model()

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]



disease_info = {

    "Tomato___Late_blight": {
        "Cause": "Fungal-like pathogen (Phytophthora infestans).",
        "Symptoms": "Large dark brown lesions on leaves and stems.",
        "Treatment": "Remove infected leaves and apply fungicides.",
        "Prevention": "Avoid overhead watering and ensure proper spacing."
    },

    "Tomato___healthy": {
        "Cause": "No disease detected.",
        "Symptoms": "Healthy green foliage.",
        "Treatment": "No treatment required.",
        "Prevention": "Continue good agricultural practices."
    },

    "Tomato___Early_blight": {
        "Cause": "Alternaria solani fungus.",
        "Symptoms": "Dark concentric spots on older leaves.",
        "Treatment": "Use fungicides and remove infected leaves.",
        "Prevention": "Crop rotation and proper sanitation."
    }

}

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(
            image,
            caption="Uploaded Leaf Image",
            width=400
        )

    with col2:

        img = image.resize((224, 224))

        img_array = np.array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = preprocess_input(
            img_array.astype(np.float32)
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )

        top3_indices = np.argsort(
            prediction[0]
        )[-3:][::-1]

        st.markdown("## Prediction Result")

        predicted_index = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        ) * 100

        disease = class_names[
            predicted_index
        ]

        display_name = disease.replace(
            "___",
            " - "
        )

        display_name = display_name.replace(
            "_",
            " "
        )

        st.success(
            f"🌿 Disease: {display_name}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Top 3 Predictions")

        for idx in top3_indices:

            class_name = class_names[idx]

            class_name = class_name.replace(
                "___",
                " - "
            )

            class_name = class_name.replace(
                "_",
                " "
            )

            confidence_score = (
                prediction[0][idx] * 100
            )

            st.write(
                f"{class_name}: {confidence_score:.2f}%"
            )

            st.progress(
                float(confidence_score) / 100
            )

        if disease in disease_info:

            st.subheader("Disease Information")

            st.write(
                f"**Cause:** {disease_info[disease]['Cause']}"
            )

            st.write(
                f"**Symptoms:** {disease_info[disease]['Symptoms']}"
            )

            st.write(
                f"**Treatment:** {disease_info[disease]['Treatment']}"
            )

            st.write(
                f"**Prevention:** {disease_info[disease]['Prevention']}"
            )