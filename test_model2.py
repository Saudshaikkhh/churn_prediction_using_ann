import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('Churnm001.h5')

# Define scaler with the same feature names for scaling tenure, MonthlyCharges, TotalCharges
scaler = MinMaxScaler()

# Features needed for the model, you can get these from your processed dataframe columns except target
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                 'PhoneService', 'MultipleLines', 'InternetService_DSL',
                 'InternetService_Fiber optic', 'InternetService_No',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract_Month-to-month',
                 'Contract_One year', 'Contract_Two year', 'PaperlessBilling',
                 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                 'MonthlyCharges', 'TotalCharges']

# To get the correct scaler parameters, we need original training data stats.
# For demonstration, you should save scaler parameters during training and load here.
# But here, I'm assuming you have the original training data loaded to fit scaler:

# Load original training data for scaler fitting (you must keep the data file)
df = pd.read_csv('customer_churn.csv')
df.drop('customerID', axis=1, inplace=True)
df = df[df.TotalCharges != ' ']
df.TotalCharges = pd.to_numeric(df.TotalCharges)
df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

yn_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in yn_columns:
    df.loc[:, col] = df[col].replace({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})

df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])

colscale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler.fit(df[colscale])

# Helper function to get user input and preprocess it
def get_user_input():
    print("Please enter the following customer details:")

    # Collecting inputs
    gender = input("Gender (Female/Male): ").strip()
    gender = 1 if gender.lower() == 'female' else 0

    senior_citizen = int(input("Senior Citizen (0 = No, 1 = Yes): ").strip())

    partner = input("Partner (Yes/No): ").strip()
    partner = 1 if partner.lower() == 'yes' else 0

    dependents = input("Dependents (Yes/No): ").strip()
    dependents = 1 if dependents.lower() == 'yes' else 0

    tenure = float(input("Tenure (months): ").strip())

    phone_service = input("Phone Service (Yes/No): ").strip()
    phone_service = 1 if phone_service.lower() == 'yes' else 0

    multiple_lines = input("Multiple Lines (Yes/No): ").strip()
    multiple_lines = 1 if multiple_lines.lower() == 'yes' else 0

    internet_service = input("Internet Service (DSL/Fiber optic/No): ").strip()

    online_security = input("Online Security (Yes/No): ").strip()
    online_security = 1 if online_security.lower() == 'yes' else 0

    online_backup = input("Online Backup (Yes/No): ").strip()
    online_backup = 1 if online_backup.lower() == 'yes' else 0

    device_protection = input("Device Protection (Yes/No): ").strip()
    device_protection = 1 if device_protection.lower() == 'yes' else 0

    tech_support = input("Tech Support (Yes/No): ").strip()
    tech_support = 1 if tech_support.lower() == 'yes' else 0

    streaming_tv = input("Streaming TV (Yes/No): ").strip()
    streaming_tv = 1 if streaming_tv.lower() == 'yes' else 0

    streaming_movies = input("Streaming Movies (Yes/No): ").strip()
    streaming_movies = 1 if streaming_movies.lower() == 'yes' else 0

    contract = input("Contract (Month-to-month/One year/Two year): ").strip()

    paperless_billing = input("Paperless Billing (Yes/No): ").strip()
    paperless_billing = 1 if paperless_billing.lower() == 'yes' else 0

    payment_method = input("Payment Method (Bank transfer (automatic)/Credit card (automatic)/Electronic check/Mailed check): ").strip()

    monthly_charges = float(input("Monthly Charges: ").strip())

    total_charges = float(input("Total Charges: ").strip())

    # Now build the feature vector with proper one-hot encoding for InternetService, Contract, PaymentMethod

    features = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        # One-hot encode InternetService
        'InternetService_DSL': 1 if internet_service.lower() == 'dsl' else 0,
        'InternetService_Fiber optic': 1 if internet_service.lower() == 'fiber optic' else 0,
        'InternetService_No': 1 if internet_service.lower() == 'no' else 0,
        # One-hot encode Contract
        'Contract_Month-to-month': 1 if contract.lower() == 'month-to-month' else 0,
        'Contract_One year': 1 if contract.lower() == 'one year' else 0,
        'Contract_Two year': 1 if contract.lower() == 'two year' else 0,
        # One-hot encode PaymentMethod
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method.lower() == 'bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method.lower() == 'credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method.lower() == 'electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method.lower() == 'mailed check' else 0,
    }

    # Create DataFrame for scaling
    df_input = pd.DataFrame([features])

    # Scale numerical columns
    df_input[colscale] = scaler.transform(df_input[colscale])

    # Reorder columns to match the model input order
    df_input = df_input[feature_names]

    return df_input.values

# Get user input features as numpy array
user_features = get_user_input()

# Predict churn probability
prediction_prob = model.predict(user_features)[0][0]

# Convert probability to binary class
prediction = "Churn" if prediction_prob > 0.5 else "No Churn"

print(f"\nPredicted probability of churn: {prediction_prob:.4f}")
print(f"Prediction: {prediction}")
