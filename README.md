# Breast Cancer Prediction Website

This project is a web application developed using **Streamlit** that allows users to predict whether a breast cancer case is benign or malignant. The application uses both Deep Learning (DL) and Machine Learning (ML) models to perform the predictions.

![Breast Cancer Prediction](https://drive.google.com/uc?id=1OxX9dezsLCo5QGQuPghkc73xl_Vi7are)

## Features

- **User-Friendly Interface**: The application provides an intuitive UI developed with Streamlit, making it easy to upload data and view predictions.
- **Multiple Prediction Models**: The application employs both a Deep Learning model and a Machine Learning model to predict breast cancer classification.
- **Real-time Prediction**: Users can upload data in real-time to receive predictions immediately.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/quyk67uet/BreastCancer_DL_ML_model.git
    cd breast-cancer-prediction
    ```

2. **Install required dependencies**:
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    streamlit run app/main.py
    ```

## Project Structure

- `app/`: Contains the main application code.
  - `main.py`: The entry point for running the Streamlit application.
  - `models/`: Contains the pre-trained models for prediction.
- `data/`: (Optional) Folder to store input data for testing.
- `requirements.txt`: List of Python dependencies required to run the application.
- `README.md`: Project documentation.

## Usage

1. **Launching the App**: Run the application using the command provided above. This will open the Streamlit app in your default web browser.

2. **Uploading Data**: You can upload a CSV file containing the relevant features for prediction. The application will process the data and display the prediction results.

3. **Viewing Predictions**: The prediction results will be shown in the app interface, indicating whether the breast cancer case is predicted as benign or malignant.

## Models

The project uses two models:

1. **Deep Learning Model**: Trained using a neural network to classify the data.
2. **Machine Learning Model**: A traditional ML model (LogisticRegression) used for comparison.




