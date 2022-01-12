import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import streamlit as st

#Loading in the model
model = tf.keras.models.load_model('saved_model/sign-language')

st.write("""
         # Simple Sign Language Hand Classification Tool
         """
         )


file = st.file_uploader("Upload the image file", type=["jpg", "png"])

def classification(img, model):



    # Create the array of the right shape to feed into the keras model

    data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.float32)

    image = img

    #image sizing

    size = (100, 100)

    image = ImageOps.fit(image, size, Image.ANTIALIAS)


    #turn the image into a numpy array

    image_array = np.asarray(image)

    # Normalize the image

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1


    # Load the image into the array

    data[0] = normalized_image_array


    # run the inference

    prediction = model.predict(data)

    return np.argmax(prediction) # return position of the highest probability

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = classification(image, )
    labels = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O', 'P','Q','R','S','space','T','U','V','W','X','Y','Z']
    st.write(labels[predictions])
