import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, BeitForImageClassification

# 1. Page Configuration
st.set_page_config(
    page_title="Fog Detection & Action Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# Custom CSS for styling, colors, and layout improvement
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .uploadedFile {
        border: 2px dashed #4f46e5;
        border-radius: 10px;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Model and Processor (Cached so it loads only once)
@st.cache_resource
def load_model_and_processor():
    model_path = r"D:\Projects\Python_projects\Machine_Learning\fog\saved_beit_model"
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = BeitForImageClassification.from_pretrained(model_path)
    model.eval()
    return processor, model

with st.spinner("✨ Loading AI model... Please wait a moment."):
    processor, model = load_model_and_processor()

# 3. Dashboard UI Header
st.markdown("""
    <div style="background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🌫️ Fog Detection & Advisory Dashboard</h1>
        <p style="font-size: 1.1rem; margin-top: 10px; opacity: 0.9;">
            Intelligent computer vision system to classify atmospheric visibility and deliver instant safety recommendations.
        </p>
    </div>
""", unsafe_allow_html=True)

# 4. Action Recommendation Mapping (The "Narrative/What to do" Engine)
recommendations = {
    "Clear": {
        "status": "🟢 Safe Conditions",
        "color": "success",
        "action": "Visibility is optimal. Normal driving and operational rules apply. Safe to proceed at standard speeds."
    },
    "Medium fog": {
        "status": "🟡 Caution Required",
        "color": "warning",
        "action": "Visibility is reduced. **Turn on low-beam headlights**, reduce speed by 10-20 km/h, and maintain a safe following distance from vehicles ahead."
    },
    "Dense fog": {
        "status": "🔴 High Hazard / Critical",
        "color": "error",
        "action": "Severe visibility restriction! **Use fog lights**, drive extremely slowly, use lane markings as a guide, or consider pulling over safely until visibility improves."
    }
}

# 5. File Uploader Component in a styled container
st.markdown("### 📂 Upload Environment Capture")
uploaded_file = st.file_uploader("Choose an environmental image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("---")
    # Display columns for layout balance
    col1, col2 = st.columns(2, gap="large")
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("🖼️ Uploaded Image Preview")
        st.image(image, use_container_width=True, caption="Source Input Image")
        
    with col2:
        st.subheader("📊 Model Inference & Analysis")
        with st.spinner("🔍 Analyzing image features..."):
            # Preprocess image and predict
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                
            # Get class label from model configuration
            label_name = model.config.id2label[predicted_class_idx]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            confidence = probabilities[predicted_class_idx].item() * 100
            
        # Display Prediction metrics inside container layout
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(label="🎯 Condition Detected", value=label_name, delta=f"{confidence:.2f}% Confidence")
        
    # 6. Dynamic Narrative & Action Box
    st.markdown("---")
    st.subheader("📋 Actionable Advisory Report")
    
    rec = recommendations.get(label_name, {
        "status": "Unknown", "color": "info", "action": "No specific advisory available."
    })
    
    if rec["color"] == "success":
        st.success(f"### {rec['status']}\n\n{rec['action']}")
    elif rec["color"] == "warning":
        st.warning(f"### {rec['status']}\n\n{rec['action']}")
    else:
        st.error(f"### {rec['status']}\n\n{rec['action']}")
        
    # Expandable view for raw probability distributions
    with st.expander("📈 View Detailed Class Probability Distributions"):
        for idx, prob in enumerate(probabilities):
            cls_name = model.config.id2label[idx]
            st.progress(int(prob.item() * 100), text=f"**{cls_name}**: {prob.item() * 100:.2f}%")