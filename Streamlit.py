import os
import streamlit as st
from PIL import Image
from gradio_client import Client, handle_file

# Client setup for remote API
client = Client("theodinproject/skin_cancer_model_resnet50v2")

# Predict disease using the external API
def predict(image_file):
    try:
        # Call the remote model API
        result = client.predict(
            image=handle_file(image_file),
            api_name="/predict"
        )

        # Parse result from API
        disease = result.get("Disease", "Unknown")
        accuracy = result.get("Accuracy", "N/A")

        return {"Disease": disease, "Accuracy": accuracy}
    except Exception as e:
        return {"error": str(e)}

# Streamlit Interface
st.title("🩺 Skin Disease Classification")
st.markdown(
    """
    Upload an image of the skin to classify the disease using a ResNet-50v2 model.
    The application connects to an advanced model hosted remotely for predictions.
    """
)

uploaded_file = st.file_uploader(
    "Upload a skin image (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Ensure the "temp" directory exists
    os.makedirs("temp", exist_ok=True)

    # Save the uploaded image temporarily for processing
    temp_file_path = "temp/temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the disease
    result = predict(temp_file_path)

    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.success(f"Disease: {result['Disease']}")
        st.info(f"Accuracy: {result['Accuracy']}")
