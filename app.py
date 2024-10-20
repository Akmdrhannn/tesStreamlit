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
    json_url = 'https://drive.google.com/uc?id=1v46pG70CKMleuil1aeGbJdqI8RutxbVX'
    weights_url = 'https://drive.google.com/uc?id=173GNh9ed4Yxib9rp_372O3Li-PALQZf6'
    
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
    # Resize gambar sesuai input model (misal: 256x256)
    gray_image = cv2.resize(gray_image, (256, 256))
    # Normalisasi nilai piksel
    gray_image = gray_image / 255.0
    # Ubah bentuk gambar menjadi 3 channel
    gray_image = np.stack([gray_image] * 3, axis=-1)  # Duplicate channel to create a 3-channel image
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
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('bg.jpg');
            background-size: cover;
        }
        .center-image {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Pewarnaan Citra Grayscale Menggunakan CGAN")
    
    # Teks yang diminta
    st.write("**_Conditional Generative Adversarial Networks_ (CGAN)** adalah metode yang menjadikan sebuah masukan atau _input_ -nya seperti sebuah kondisi. Merujuk pada _Conditional_, kondisi tersebut dapat digunakan terhadap _generator_ atau _discriminator_ sesuai kebutuhan pengguna sehingga CGAN dapat menghasilkan data buatan yang diinginkan menggunakan kondisi yang spesifik atau telah ditentukan sebelumnya. Pada penelitian ini, kondisi tersebut ialah citra _grayscale_ yang menjadi syarat dalam pelatihan model untuk menghasilkan pewarnaan.")
    
    # Tampilkan gambar di tengah
    st.markdown('<div class="center-image"><img src="gambaranUmum.png" width="500"></div>', unsafe_allow_html=True)
    
    st.write("Silahkan upload citra atau gambar grayscale yang akan diwarnai oleh model GAN.")
    
    # Upload gambar
    uploaded_file = st.file_uploader("Pilih gambar grayscale", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca file gambar yang diupload
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption='Citra Grayscale', use_column_width=True)
        
        # Preprocess gambar dan warnai menggunakan model
        gray_image = preprocess_image(image)
        model = load_model()
        color_image = colorize_image(model, gray_image)
        
        # Tampilkan hasil gambar berwarna
        st.image(color_image, caption='Hasil Pewarnaan', use_column_width=True)

if __name__ == '__main__':
    main()
