# 🩺 Skin Disease Classification App

This AI-powered **Skin Disease Classification App** leverages a **ResNet-50 v2** model trained on a custom dataset to identify potential skin conditions from uploaded images. Hosted on **Streamlit**, the app connects to a **Hugging Face Space API** for real-time predictions.

## 🌟 Features
- **Image Upload**: Upload images (JPG, JPEG, PNG) of skin lesions for classification.
- **AI-powered Predictions**: Classifies images into five categories:
  - Burn Skin
  - Healthy Skin
  - Malignant
  - Non-Cancerous
  - Non-Skin
- **Real-time Results**: Displays the predicted disease category and confidence score.
- **User-friendly Interface**: A responsive and intuitive UI built with Streamlit.

## 🚀 Live Demo
Access the app here: [Skin Disease Classification App](https://smart-healthcare.streamlit.app/)  

## 🛠️ Technologies Used
- **[Streamlit](https://streamlit.io/)**: Interactive web application framework.
- **[ResNet-50 v2](https://arxiv.org/abs/1603.05027)**: Advanced deep learning model for image classification.
- **[Hugging Face Spaces](https://huggingface.co/spaces)**: Model hosting for API integration.
- **[Python](https://www.python.org/)**: Core language for development.
- **[TensorFlow](https://www.tensorflow.org/)**: Training the ResNet-50 v2 model.

## 📋 How It Works
1. **Upload an Image**: Use the file uploader to provide a skin lesion image.
2. **Remote Processing**: The uploaded image is sent to the hosted ResNet-50 v2 model API for analysis.
3. **Prediction Results**: The app displays the classified disease and confidence score.

## 🖥️ Installation (For Local Use)
Follow these steps to set up the app locally:

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

5. **Open the App**:
   Navigate to `http://localhost:8501` in your browser.

## 📂 Project Structure
```
.
├── Streamlit.py           # Main Streamlit application
├── requirements.txt       # Dependencies
├── utils\train_model.py   # Utility scripts (e.g., model training)
├── temp/                  # Temporary folder for images
├── README.md              # Project documentation
└── data/                  # Dataset splits (train, validation, test)
```

## ⚠️ Disclaimer
This app is intended for **educational purposes only**. It is not a substitute for professional medical advice. Please consult a certified medical professional for accurate diagnosis and treatment.

## 📞 Contact
For feedback or inquiries:
- **Name**: Manish Borikar  
- **Email**: [manishborikar@proton.me](mailto:manishborikar@proton.me)  
- **GitHub**: [manishborikar92](https://github.com/manishborikar92)

## 🏅 Acknowledgments
- **Hugging Face** for providing the model hosting platform.
- **Streamlit** for simplifying web app development.
- **TensorFlow** for training the ResNet-50 v2 model.