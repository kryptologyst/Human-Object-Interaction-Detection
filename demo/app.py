"""Streamlit demo for HOI detection."""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hoi_detection import HOIDetector
from hoi_detection.utils import visualize_hoi, setup_device, set_seed


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Human-Object Interaction Detection",
        page_icon="🤝",
        layout="wide"
    )
    
    st.title("🤝 Human-Object Interaction Detection")
    st.markdown("Detect and visualize human-object interactions in images")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Device selection
    device_options = ["auto", "cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_options.append("mps")
    
    device = st.sidebar.selectbox("Device", device_options, index=0)
    
    # Score threshold
    score_threshold = st.sidebar.slider(
        "Score Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.1
    )
    
    # Backend selection
    backend = st.sidebar.selectbox(
        "Backend", 
        ["detectron2", "custom"], 
        index=0
    )
    
    # Initialize detector
    @st.cache_resource
    def load_detector(device, score_threshold, backend):
        """Load HOI detector with caching."""
        try:
            detector = HOIDetector(
                device=device,
                score_threshold=score_threshold,
                backend=backend
            )
            return detector
        except Exception as e:
            st.error(f"Failed to load detector: {e}")
            return None
    
    detector = load_detector(device, score_threshold, backend)
    
    if detector is None:
        st.error("Failed to initialize detector. Please check your configuration.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect human-object interactions"
        )
        
        # Sample images
        st.subheader("Sample Images")
        sample_images = [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
            "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400",
            "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400"
        ]
        
        for i, sample_url in enumerate(sample_images):
            if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                # Download sample image
                import requests
                response = requests.get(sample_url)
                if response.status_code == 200:
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(response.content)
                        uploaded_file = tmp.name
        
        if uploaded_file is not None:
            # Load image
            if isinstance(uploaded_file, str):
                # Sample image path
                image = cv2.imread(uploaded_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Uploaded file
                image = Image.open(uploaded_file)
                image = np.array(image)
            
            st.image(image, caption="Input Image", use_column_width=True)
            
            # Detection button
            if st.button("Detect HOI", type="primary"):
                with st.spinner("Detecting human-object interactions..."):
                    try:
                        # Run detection
                        results = detector.predict(image)
                        
                        # Visualize results
                        vis_image = detector.visualize(image, results)
                        
                        # Display results
                        st.success("Detection completed!")
                        
                        with col2:
                            st.header("Detection Results")
                            st.image(vis_image, caption="HOI Detection Results", use_column_width=True)
                            
                            # Display detection info
                            if results['instances'] is not None:
                                instances = results['instances']
                                st.subheader("Detection Summary")
                                st.write(f"**Number of objects detected:** {len(instances)}")
                                
                                # Object classes
                                if hasattr(instances, 'pred_classes'):
                                    classes = instances.pred_classes.cpu().numpy()
                                    scores = instances.scores.cpu().numpy()
                                    
                                    st.subheader("Detected Objects")
                                    for i, (cls, score) in enumerate(zip(classes, scores)):
                                        st.write(f"Object {i+1}: Class {cls}, Score: {score:.3f}")
                                
                                # Interactions
                                if results['interactions'] is not None:
                                    interactions = results['interactions']
                                    st.subheader("Detected Interactions")
                                    st.write(f"Interaction matrix shape: {interactions.shape}")
                                    
                                    # Find strongest interactions
                                    interaction_scores = interactions.max(dim=-1)[0]
                                    if interaction_scores.max() > 0.1:  # Threshold for display
                                        st.write("Strong interactions detected!")
                                
                            else:
                                st.warning("No objects detected in the image.")
                        
                    except Exception as e:
                        st.error(f"Detection failed: {e}")
                        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Human-Object Interaction Detection** | "
        "Built with PyTorch, Detectron2, and Streamlit"
    )


if __name__ == "__main__":
    main()
