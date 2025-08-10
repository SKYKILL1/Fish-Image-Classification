import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa

# Set page config with attractive styling
st.set_page_config(
    page_title="AquaVision: Fish Species Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #0066cc, #00ccff);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .progress-bar {
        height: 10px;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 5px;
        margin-top: 10px;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
</style>
""", unsafe_allow_html=True)

# Load model with custom objects
@st.cache_resource
def load_model():
    custom_objects = {
        'F1Score': tfa.metrics.F1Score,
        'Addons>F1Score': tfa.metrics.F1Score  # Handle both naming formats
    }
    try:
        return tf.keras.models.load_model(
            './Models/model_MobileNet.h5',
            custom_objects=custom_objects
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Class labels (replace with your actual class names)
class_names = [
    'Animal Fish', 'Bass', 'Black Sea Sprat', 'Gilt Head Bream', 
    'Hourse Mackerel', 'Red Mullet', 'Red Sea Bream', 
    'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout'
]

# Preprocess image
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Main app
def main():
    # Header with gradient
    st.markdown("""
    <div class="header">
        <h1>🐠 AquaVision</h1>
        <h3>AI-Powered Fish Species Classification</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3079/3079158.png", width=100)
        st.title("About")
        st.info("""
        This app identifies fish species using deep learning. 
        Upload an image or use your camera to classify fish.
        """)
        
        st.markdown("---")
        st.subheader("How to use:")
        st.write("1. Upload a fish image or use camera")
        st.write("2. Wait for processing")
        st.write("3. View results and confidence")
        
        st.markdown("---")
        st.write("Developed by [Your Name]")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload or Capture Image")
        img_source = st.radio("Select input source:", 
                             ("Upload an image", "Use camera"))
        
        img_file = None
        if img_source == "Upload an image":
            img_file = st.file_uploader("Choose a fish image...", 
                                      type=["jpg", "jpeg", "png"])
        else:
            img_file = st.camera_input("Take a picture of fish")
    
    with col2:
        st.subheader("Classification Results")
        
        if img_file is not None and model is not None:
            # Display image
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Image", width=300)
            
            # Process and predict
            with st.spinner("Analyzing fish species..."):
                processed_img = preprocess_image(img)
                predictions = model.predict(processed_img)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
            
            # Display results with animation
            st.success("Analysis Complete!")
            
            # Result card
            st.markdown(f"""
            <div class="result-card fade-in">
                <h3>Prediction: {class_names[predicted_class]}</h3>
                <p>Confidence: {confidence:.2f}%</p>
                <div class="progress-bar" style="width:{confidence}%"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence breakdown
            st.subheader("Confidence Breakdown")
            for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
                st.markdown(f"""
                <div class="fade-in">
                    <p>{class_name}: {prob*100:.1f}%</p>
                    <progress value="{prob}" max="1" style="width:100%"></progress>
                </div>
                """, unsafe_allow_html=True)
            
            # Fun fact about the fish
            fun_facts = {
                'Bass': "Did you know? Bass can live up to 20 years!",
                'Trout': "Trout have a special adipose fin that few other fish possess.",
            }
            if class_names[predicted_class] in fun_facts:
                st.info(f"**Fun Fact:** {fun_facts[class_names[predicted_class]]}")

if __name__ == "__main__":
    main()