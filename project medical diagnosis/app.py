import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image

# Load Model
model = load_model('models/pneumonia_model.h5')

IMG_SIZE = 224

classes = ['NORMAL', 'PNEUMONIA']

# Prediction Function
def predict_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    probability = prediction[0][0]

    if probability > 0.5:
        result = classes[1]
        confidence = probability
    else:
        result = classes[0]
        confidence = 1 - probability

    return result, confidence

# Streamlit UI
st.set_page_config(page_title='Medical Diagnoser', layout='centered')
st.title('AI Medical Diagnoser')
st.subheader('Chest X-Ray Pneumonia Detection')
uploaded_file = st.file_uploader(
    'Upload Chest X-Ray Image',
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    st.image(img, caption='Uploaded X-Ray', use_container_width=True)
    if st.button('Analyze X-Ray'):
        with st.spinner('Analyzing Image...'):
            result, confidence = predict_image(img)
            st.success(f'Prediction: {result}')
            st.info(f'Confidence: {confidence * 100:.2f}%')
            if result == 'PNEUMONIA':
                st.error('Possible Pneumonia Detected')
            else:
                st.success('Lungs appear Normal')