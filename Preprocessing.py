import streamlit as st
from PIL import Image
img = Image.open("C:/Users/GARAO/Pictures/Screenshots/Screenshot (14).png")
st.image(img, caption="Sample Image")