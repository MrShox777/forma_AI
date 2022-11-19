import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath


st.title("Transportni klasfikatsiya qiluvchi model")

file = st.file_uploader("saved PNJ", type=["pnj", 'jpeg', "gif", "svg", "jpg" ])

img = PILImage.create(file)

model = load_learner('transport_model.pkl')

prediction = model.predict(img)
st.success(prediction)