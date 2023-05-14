import streamlit as st
import tensorflow as tf
import pandas as pd
from streamlit_option_menu import option_menu
from PIL import Image, ImageOps
import numpy as np

#Inorder to ignore the warning when we upload an image
st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
    # Medical Image Analysis and Classification
    """
    )

selected = option_menu(
    menu_title = None,
    options = ["Blood Cell Classifier","Breast Cancer"],
    icons = ["droplet","book"],
    menu_icon = "cast",
    default_index = 0,
    orientation = "horizontal",
)


if selected == "Blood Cell Classifier":
    st.title(f" {selected}")

    def load_model():
        model = tf.keras.models.load_model(r"model_blood.h5")
        return model

    model= load_model()

    st.write("You can either upload an image file or open webcam for taking a pic:")

    selected = option_menu(
        menu_title = None,
        options = ["Browse File","Open Webcam"],
        icons = ["file-arrow-up-fill","camera"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal",
    )

    if selected == "Browse File":

        file = st.file_uploader("Please upload the sample image", type=["jpg","png"])
    
        generate_pred = st.button("Predict")

        def import_and_predict(image_data, model):
    
            size=(28,28)
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            img = np.asarray(image)
            img_reshape = img[np.newaxis,...]
            prediction = model.predict(img_reshape)
            return prediction

        if generate_pred and file != None:
            image = Image.open(file)
            with st.expander('image',expanded=True):
                st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            labels = ['class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8']
            st.title("The image belongs to {}".format(labels[np.argmax(prediction)]))
        if generate_pred and file == None:
            st.text("Please upload an image File!")

    if selected == "Open Webcam":
        imageCaptured = st.camera_input("Capture Image",key="firstCamera" )
        st.image(imageCaptured)
        generate_pred = st.button("Predict")

        def import_and_predict(image_data, model):
    
            size=(28,28)
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            img = np.asarray(image)
            img_reshape = img[np.newaxis,...]
            prediction = model.predict(img_reshape)
            return prediction

        if generate_pred:
            image = Image.open(imageCaptured)
            with st.expander('image',expanded=True):
                st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            labels = ['class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8']
            st.title("The image belongs to  {}".format(labels[np.argmax(prediction)]))
        
        


if selected == "Breast Cancer":

    st.title(f" {selected}")

    #Inorder to ignore the warning when we upload an image
    st.set_option('deprecation.showfileUploaderEncoding', False)

    def load_model():
        model = tf.keras.models.load_model(r"model_breast.h5")
        return model

    model= load_model()

    st.write("You can either upload an image file or open webcam for taking a pic:")

    selected = option_menu(
        menu_title = None,
        options = ["Browse File","Open Webcam"],
        icons = ["file-arrow-up-fill","camera"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal",
    )

    if selected == "Browse File":

        file = st.file_uploader("Please upload the sample image", type=["jpg","png"])
    
        generate_pred = st.button("Predict")

        def import_and_predict(image_data, model):
    
            size=(28,28)
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            img = np.asarray(image)
            img_reshape = img[np.newaxis,...]

            prediction = (model.predict(img_reshape)>=0.96).astype('int')
            return prediction

        if generate_pred and file != None:
            image = Image.open(file)
            with st.expander('image',expanded=True):
                st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            labels = ['Patient has Positive signs of Breast Cancer. Refer to a doctor.','Patient has no signs of Breast Cancer']
            st.title(" {}".format(labels[np.argmax(prediction)]))
        if generate_pred and file == None:
            st.text("Please upload an image File!")

    if selected == "Open Webcam":
        imageCaptured = st.camera_input("Capture Image",key="firstCamera" )
        st.image(imageCaptured)
        generate_pred = st.button("Predict")

        def import_and_predict(image_data, model):
    
            size=(28,28)
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            img = np.asarray(image) 
            img_reshape = img[np.newaxis,...]

            prediction = (model.predict(img_reshape)>=0.96).astype('int')
            return prediction

        if generate_pred:
            image = Image.open(imageCaptured)
            with st.expander('image',expanded=True):
                st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            labels = ['Patient has Positive signs of Breast Cancer. Refer to a doctor.','Patient has no signs of Breast Cancer']

            if prediction == 0:
                st.title(" {}".format(labels[prediction]))
            else:
                st.title(" {}".format(labels[prediction]))

            



