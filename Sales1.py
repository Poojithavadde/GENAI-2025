import streamlit as st


# Center the title using 3 columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("Sales Dashboard")

st.write(" ")  # Spacer

# Centered 4-column metric layout with padding columns on each side
col1, space1, col2, space2, col3, space3, col4 = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1])
with col1:
    st.subheader("Total Sales")
    

with col2:
    st.subheader("Average Value")
    

with col3:
    st.subheader("Total Quality")
    

with col4:
    st.subheader("Total Profit")
    

# Spacer
st.write("")
st.write("")


# Dummy text also centered using 3 columns
col1, space1, col2, space2, col3, space3,  = st.columns([1, 0.2, 1, 0.2, 1, 0.2, ])
with col2:
    st.markdown("### Dummy text")
    st.markdown("### Dummy text")