import os

os.system("pip install opencv-python-headless streamlit numpy scikit-image pillow")
import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io

def preprocess_signature(image):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    return binary

def align_signatures(img1, img2):
    # Resize images to same size
    target_size = (300, 200)
    img1_resized = cv2.resize(img1, target_size)
    img2_resized = cv2.resize(img2, target_size)
    
    return img1_resized, img2_resized

def calculate_similarity(img1, img2):
    # Calculate SSIM between the two images
    similarity_score = ssim(img1, img2)
    return similarity_score * 100  # Convert to percentage

def main():
    st.title("Signature Comparison Tool")
    
    # File uploaders for signatures
    sig1 = st.file_uploader("Upload First Signature", type=['png', 'jpg', 'jpeg'])
    sig2 = st.file_uploader("Upload Second Signature", type=['png', 'jpg', 'jpeg'])
    
    if sig1 and sig2:
        # Convert uploaded files to opencv format
        file_bytes1 = np.asarray(bytearray(sig1.read()), dtype=np.uint8)
        file_bytes2 = np.asarray(bytearray(sig2.read()), dtype=np.uint8)
        
        img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
        
        # Preprocess signatures
        proc_img1 = preprocess_signature(img1)
        proc_img2 = preprocess_signature(img2)
        
        # Align signatures
        aligned_img1, aligned_img2 = align_signatures(proc_img1, proc_img2)
        
        # Calculate similarity
        match_percentage = calculate_similarity(aligned_img1, aligned_img2)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(aligned_img1, caption="Signature 1", use_column_width=True)
        
        with col2:
            st.image(aligned_img2, caption="Signature 2", use_column_width=True)
        
        st.write(f"Matching Percentage: {match_percentage:.2f}%")
        
        if match_percentage > 80:
            st.success("Signatures are highly similar!")
        elif match_percentage > 60:
            st.warning("Signatures have moderate similarity")
        else:
            st.error("Signatures are significantly different")

if __name__ == "__main__":
    main() 
