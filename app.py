import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import gdown

model_url = 'https://raw.githubusercontent.com/JuliaMarieAbad/ABAD_LEOPANDO_Final-Exam-Model-Deployment-in-the-Cloud/main/app.py'
model_path = 'cat_dog_classifier.h5'

@st.cache(hash_funcs={type(lambda: 0): id})  
def load_model():
    try:
        gdown.download(model_url, model_path, quiet=False)
        model = tf.keras.models.load_model(model_path)
        return model
    except gdown.exceptions.FileURLRetrievalError as e:
        st.error(f"Error retrieving the model file: {e}")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
    return None

model = load_model()

if model is not None:
    class_names = ['cat', 'dog']

    def predict(image):
        image = image.resize((256, 256))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predicted_class = class_names[int(np.round(prediction[0]))]
        return predicted_class

    st.title("Cat and Dog Image Classification App")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = predict(image)
        st.write(f"Prediction: {label}")
