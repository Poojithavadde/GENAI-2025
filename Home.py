import streamlit as st
st.write("Hello World")

name = st.text_input("Enter your name:")
st.write("Hello", name)


if st.button("Click Me"):
    st.write("Button clicked!")


age = st.slider("Select your age", 1, 100)
st.write("Your age is", age)



if st.checkbox("Show message"):
    st.write("Checkbox is checked!")


import pandas as pd
df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [85, 90]})
st.dataframe(df)



st.metric(label="Temperature", value="32°C", delta="-1.2°C")


st.radio("Choose one", ["A", "B", "C"])

