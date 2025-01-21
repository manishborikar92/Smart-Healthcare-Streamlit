# 🩺 Skin Disease Classification App

This is an AI-powered **Skin Disease Classification App** that uses a **ResNet-50 v2** model to identify potential skin conditions from uploaded images. The app is hosted on **Streamlit** and connects to a remote API for real-time predictions.

## 🌟 Features
- **Upload Images**: Easily upload skin images (JPG, JPEG, PNG) to classify potential skin diseases.
- **AI-powered Predictions**: Uses a pre-trained **ResNet-50 v2** model to classify images into five categories:
  - Burn Skin
  - Healthy Skin
  - Malignant
  - Non-Cancerous
  - Non-Skin
- **Real-time Results**: Displays the predicted disease name along with the confidence score.
- **User-friendly Interface**: Powered by Streamlit, offering an intuitive and interactive experience.

## 🚀 Live Demo
You can access the app here: [Skin Disease Classification App](https://smart-healthcare.streamlit.app/)  

## 🛠️ Built With
- **[Streamlit](https://streamlit.io/)**: For the web app interface.
- **[ResNet-50 v2](https://arxiv.org/abs/1603.05027)**: A state-of-the-art deep learning model for image classification.
- **[Hugging Face Space](https://huggingface.co/spaces)**: Hosting the pre-trained model as an API.
- **[Python](https://www.python.org/)**: Backend logic and integration.

## 📋 How It Works
1. **Upload**: Upload an image of the skin lesion or area of interest.
2. **Processing**: The image is sent to a remote ResNet-50 v2 model API for analysis.
3. **Results**: The app displays the predicted disease category and its confidence score.

## 🖥️ Installation (For Local Use)
To run the app locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/manishborikar92/Smart-Healthcare-Streamlit.git
   cd Smart-Healthcare-Streamlit
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   ```bash
   streamlit run Streamlit.py
   ```

5. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

## 📂 Project Structure
```
.
├── Streamlit.py        # Main Streamlit application
├── requirements.txt    # Python dependencies
├── temp/               # Temporary folder for uploaded images
├── README.md           # Project documentation
```

## ⚠️ Disclaimer
This app is for **educational purposes only** and should not be used for medical diagnosis or treatment. Always consult a certified medical professional for health-related concerns.

## 📞 Contact
For any queries or suggestions, feel free to reach out:
- **Name**: Manish Borikar  
- **Email**: manishborikar@proton.me  
- **GitHub**: [manishborikar92](https://github.com/manishborikar92)

## 🏅 Acknowledgments
- **Hugging Face** for hosting the ResNet-50 v2 model.
- **Streamlit** for providing an excellent framework for interactive applications.
- **ResNet-50 v2 Authors** for their contribution to deep learning research.