import streamlit as st
import requests

# Define the Rust API endpoint URL
API_URL = "http://localhost:8083/api/chat"

# Streamlit app title
st.title("LLM Inference API in Rust")

# Input prompt from the user
prompt = st.text_area("Enter your prompt:")

# Function to generate a response
def generate_response(prompt):
    try:
        response = requests.post(API_URL, json={"prompt": prompt})
        if response.status_code == 200:
            result = response.json().get("response")
            st.success(f"Response: {result}")
        else:
            st.error("Error: Failed to generate a response.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")

# Submit button to generate response
if st.button("Generate Response"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        generate_response(prompt)
