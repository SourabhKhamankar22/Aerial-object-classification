import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Aerial Object Detection", page_icon="🚁", layout="wide")

# 2. Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083163.png", width=100)
    st.title("About the Project")
    st.info("""
    Data Science Capstone

    This deep learning application classifies aerial objects into:
    * 🦅 **Birds**
    * 🛸 **Drones**

    **Architecture:** MobileNetV2 (Transfer Learning)  
    **Accuracy:** 97.21%
    """)
    st.divider()
    st.caption("Developed by Sourabh Khamankar")

# 3. Main UI
st.title(" Aerial Object Classifier ")
st.markdown("Upload an image or use your camera to classify the object.")

# 4. Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_tl_model.h5", compile=False)

model = load_model()

# ✅ Explicit class mapping 
CLASS_INDICES = {'bird': 0, 'drone': 1}

# 5. Input Tabs
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take a Photo"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

with tab2:
    camera_file = st.camera_input("Take a picture")

final_image_file = uploaded_file if uploaded_file else camera_file

# 6. Prediction Pipeline
if final_image_file is not None:
    
    try:
        image = Image.open(final_image_file)
    except:
        st.error("Invalid or corrupted image.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("AI Analysis")

        # Preprocessing
        image = image.convert('RGB')
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        with st.spinner("Analyzing image..."):
            prediction = model.predict(img_array)

        #  Handle Binary Output
        if prediction.shape[1] == 1:
            prob = prediction[0][0]  # drone probability

            bird_conf = (1 - prob) * 100
            drone_conf = prob * 100

        else:
            # Multi-class (future-proof)
            probs = prediction[0]
            bird_conf = probs[CLASS_INDICES['bird']] * 100
            drone_conf = probs[CLASS_INDICES['drone']] * 100

        # Determine class
        if drone_conf > bird_conf:
            predicted_class = "Drone 🛸"
            final_conf = drone_conf
            st.error(f"### Classification: {predicted_class}")
        else:
            predicted_class = "Bird 🦅"
            final_conf = bird_conf
            st.success(f"### Classification: {predicted_class}")

        # Confidence
        st.metric("Winning Confidence", f"{final_conf:.2f}%")

        # Low confidence warning
        if final_conf < 70:
            st.warning("⚠️ Low confidence prediction. Try a clearer image.")

        st.divider()

        # Breakdown
        st.write("**Probability Breakdown:**")

        st.write(f"🦅 Bird: {bird_conf:.2f}%")
        st.progress(int(bird_conf))

        st.write(f"🛸 Drone: {drone_conf:.2f}%")
        st.progress(int(drone_conf))

        # Bar chart
        chart_data = {
            "Bird": bird_conf,
            "Drone": drone_conf
        }
        st.bar_chart(chart_data)

# 7. Footer
st.markdown("---")
st.caption("Built with TensorFlow & Streamlit | © 2026")