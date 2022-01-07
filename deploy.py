import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import streamlit as st

#Loading in the model
model = tf.keras.models.load_model('saved_model/sign-language/saved_model.pb')

st.write("""
         # Simple Sign Language Hand Classification Tool
         """
         )


file = st.file_uploader("Upload the image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    
        size = (100,100)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img_resize = (cv2.resize(img, dsize=(100, 100),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    st.write(prediction)