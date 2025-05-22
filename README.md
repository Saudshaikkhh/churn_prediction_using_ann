# 📊 Customer Churn Prediction System

A comprehensive machine learning solution for predicting customer churn using deep learning (TensorFlow/Keras) and an interactive web interface built with Streamlit.

---

## 🎯 Overview

This project predicts whether a customer will churn (leave the service) based on their demographics, account details, and service usage patterns. It includes:

- Data preprocessing
- Deep learning model training
- Streamlit-based web app
- Data visualization and performance tracking

---

## 🌟 Features

- 🔄 **Data Preprocessing Pipeline**  
  Automated data cleaning, encoding, and scaling

- 🧠 **Deep Learning Model**  
  Binary classifier using TensorFlow/Keras

- 💻 **Interactive Web Interface**  
  Built with Streamlit for real-time predictions

- 📈 **Visualizations**  
  Explore churn patterns and insights with charts

- ⚙️ **Real-time Predictions**  
  Instant churn probability calculation

- 📊 **Model Evaluation**  
  Accuracy tracking and binary classification metrics

---

## 📁 Project Structure
```bash
 customer-churn-prediction/
├── data_processing.py # Data preprocessing and model training
├── streamlit_app.py # Streamlit web application
├── customer_churn.csv # Dataset (from Kaggle)
├── Churnm001.h5 # Trained model file
├── README.md # Project documentation
└── requirements.txt # Python dependencies  
```
---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset from Kaggle, containing:

- **Demographics**: Gender, Partner, Dependents  
- **Account Info**: Contract, Payment Method, Tenure  
- **Service Use**: Internet, Security, Support services  
- **Billing**: Monthly and Total Charges, Paperless billing  
- **Target**: `Churn` (Yes/No)

🔗 **Dataset Link**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🛠️ Installation & Setup

### ✅ Prerequisites

- Python 3.7+
- pip

### 📥 Step 1: Clone the Repository
```bash
git clone <repository-url>
cd customer-churn-prediction
```

### 📦 Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### 📁 Step 3: Download and Prepare Dataset
- Visit the Kaggle dataset page
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Rename it to `customer_churn.csv`
- Place it in the root project directory

### 🔧 Step 4: Train the Model
```bash
python data_processing.py
```
This will:
- Load and preprocess the data
- Train the deep learning model
- Save the trained model as `Churnm001.h5`

### 🚀 Step 5: Run the Web App
```bash
streamlit run streamlit_app.py
```
## 🧠 Model Architecture
- Input Layer: 26 features
- Hidden Layer: 26 neurons, ReLU activation
- Output Layer: 1 neuron, Sigmoid activation
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Epochs: 500

## 🔄 Data Preprocessing
### 🧹 Data Cleaning
- Remove customerID
- Handle missing TotalCharges values
- Convert strings to numeric form
### 🛠️ Feature Engineering
- Binary encoding for Yes/No fields
- One-hot encoding for multi-category fields
- MinMax scaling of numeric features
### 🔢 Transformation
- Gender encoding: Female = 1, Male = 0
- Dummy variables for categorical data

## 🙏 Acknowledgments
- Kaggle for the dataset
- TensorFlow/Keras for the ML framework
- Streamlit for the web interface
- Python open-source community

## 📞 Contact
For questions, suggestions, or collaboration:
- Name: Your Name
- Email: your.email@example.com
- LinkedIn: Your LinkedIn

Happy Predicting! 🚀