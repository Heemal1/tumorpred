import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model_path = 'tumor_detection_1.h5'  # Replace with your actual path
model = load_model(model_path)

# Define class labels
class_labels = {
   0: 'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1',
   1: 'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+',
   2: 'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2',
   3: 'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1',
   4: 'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+',
   5: 'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2',
   6: 'NORMAL T1',
   7: 'NORMAL T2',
   8: 'Neurocitoma (Central - Intraventricular, Extraventricular) T1',
   9: 'Neurocitoma (Central - Intraventricular, Extraventricular) T1C+',
   10: 'Neurocitoma (Central - Intraventricular, Extraventricular) T2',
   11: 'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1',
   12: 'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+',
   13: 'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2',
   14: 'Schwannoma (Acustico, Vestibular - Trigeminal) T1',
   15: 'Schwannoma (Acustico, Vestibular - Trigeminal) T1C+',
   16: 'Schwannoma (Acustico, Vestibular - Trigeminal) T2'
}

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict(image_file):
    img_array = preprocess_image(image_file)
    prediction = model.predict(img_array)
    return prediction

# Custom CSS for background and text color
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://e0.pxfuel.com/wallpapers/906/952/desktop-wallpaper-graphy-background-best-ultra-750909-ssoflx-for-your-mobile-tablet-explore-best-websites-3840x2160-nature-grapher.jpg");
        background-size: cover;
    }
    .predicted-text {
        color: white;
        font-size: 20px;
        background-color:green;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title('Brain Tumor Detection')
st.text('Upload an MRI scan image for prediction')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded MRI scan', use_column_width=True)

    # Make a prediction
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            prediction = predict(uploaded_file)
            predicted_class = np.argmax(prediction)
            st.markdown(f'<p class="predicted-text">Predicted Class: {class_labels[predicted_class]}</p>', unsafe_allow_html=True)
