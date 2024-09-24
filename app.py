import streamlit as st
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import gdown  # Digunakan untuk download file dari Google Drive
import os

# Fungsi untuk mendownload file dari Google Drive
def download_file_from_google_drive(url, output):
    if not os.path.exists(output):  # Download hanya jika file belum ada
        gdown.download(url, output, quiet=False)
    else:
        st.write(f"{output} sudah ada, tidak perlu download ulang.")

# Load model generator
@st.cache_resource
def load_model():
    # Download model JSON dan weights dari Google Drive
    json_url = 'https://drive.google.com/uc?id=1hLA7-131qfIqYZGRYusTZWjUiUiGMw3L'
    weights_url = 'https://drive.google.com/uc?id=1d33hDWk4HELbpfBwWDzP1FwxQ0INZbJG'
    
    download_file_from_google_drive(json_url, 'modelGenerator.json')
    download_file_from_google_drive(weights_url, 'modelGen.weights.h5')
    
    # Load model dari JSON dan weights
    with open('modelGenerator.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('modelGen.weights.h5')
    return model

# Preprocess image untuk input ke model
def preprocess_image(image):
    # Ubah citra menjadi grayscale jika belum
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize gambar sesuai input model (misal: 150x150)
    gray_image = cv2.resize(gray_image, (256, 256))
    # Normalisasi nilai piksel
    gray_image = gray_image / 255.0
    # Ubah bentuk gambar sesuai input model (batch_size, height, width, channels)
    gray_image = gray_image.reshape(1, 256, 256, 3)
    return gray_image

# Proses pewarnaan citra menggunakan model generator
def colorize_image(model, gray_image):
    # Prediksi citra berwarna
    color_image = model.predict(gray_image)
    # Post-processing hasil prediksi
    color_image = color_image[0] * 255
    color_image = np.clip(color_image, 0, 255).astype('uint8')
    return color_image

# Streamlit UI
def main():
    st.title("Image Colorization with GAN")

    st.write("Upload a grayscale image and see it colorized by the GAN model.")
    
    # Upload gambar
    uploaded_file = st.file_uploader("Choose a grayscale image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca file gambar yang diupload
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Original Grayscale Image', use_column_width=True)
        
        # Preprocess gambar dan warnai menggunakan model
        gray_image = preprocess_image(image)
        model = load_model()
        color_image = colorize_image(model, gray_image)
        
        # Tampilkan hasil gambar berwarna
        st.image(color_image, caption='Colorized Image', use_column_width=True)

if __name__ == '__main__':
    main()
