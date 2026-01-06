import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set page config
st.set_page_config(page_title="Oil Spill Detection", layout="wide")

st.title("Oil Spill Detection from Satellite Imagery")
st.write("Upload a satellite image to detect oil spills using the U-Net model.")

# Load Model
@st.cache_resource
def load_model():
    model_path = 'models/best_model.keras'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Detect Oil Spill"):
        if model is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            with st.spinner("Analyzing..."):
                # Preprocess
                img_array = np.array(image.resize((256, 256)))
                img_input = np.expand_dims(img_array, axis=0) / 255.0

                # Predict
                prediction = model.predict(img_input)
                mask = (prediction[0] > 0.5).astype(np.uint8) * 255 # Binarize and scale to 0-255

                # Resize mask back to original image size for display/overlay
                mask_img = Image.fromarray(mask.squeeze(), mode='L').resize(image.size)
                
                # Create overlay
                overlay = Image.blend(image, mask_img.convert('RGB'), alpha=0.5)

                
                # Check if any oil spill is detected (thresholding to avoid noise)
                spill_pixels = np.count_nonzero(mask)
                total_pixels = mask.size
                spill_percentage = (spill_pixels / total_pixels) * 100
                
                # Threshold: if more than 0.1% or 50 pixels are detected (tunable)
                if spill_pixels > 50:
                    st.error(f"⚠️ Oil Spill Detected! ({spill_percentage:.2f}% of area)")
                else:
                    st.success("✅ No Oil Spill Detected.")

                with col2:
                    st.subheader("Detected Spill Mask")
                    st.image(mask_img, use_container_width=True, clamp=True)
                
                st.subheader("Overlay")
                st.image(overlay, use_container_width=True)

st.sidebar.info("This application uses a U-Net deep learning model trained on satellite imagery to segment oil spills.")
