import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('Churnm001.h5')

# Define feature names and scaler
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                 'PhoneService', 'MultipleLines', 'InternetService_DSL',
                 'InternetService_Fiber optic', 'InternetService_No',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract_Month-to-month',
                 'Contract_One year', 'Contract_Two year', 'PaperlessBilling',
                 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                 'MonthlyCharges', 'TotalCharges']

scaler = MinMaxScaler()

# Load and preprocess data for scaler fitting
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
    df[col] = df[col].replace({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})

df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])
scaler.fit(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

def get_valid_input(prompt, validation_func, error_msg):
    """Generic input validator with retry logic"""
    while True:
        user_input = input(prompt).strip()
        if validation_func(user_input):
            return user_input
        print(f"\033[31m{error_msg}\033[0m")  # Red error message

def validate_yes_no(input):
    return input.lower() in ['yes', 'no', 'y', 'n']

def validate_gender(input):
    return input.lower() in ['male', 'female', 'm', 'f']

def validate_internet(input):
    return input.lower() in ['dsl', 'fiber optic', 'no']

def validate_contract(input):
    return input.lower() in ['month-to-month', 'one year', 'two year']

def validate_payment(input):
    return input.lower() in ['bank transfer (automatic)', 'credit card (automatic)', 
                           'electronic check', 'mailed check']

def validate_number(input, min_val=0):
    try:
        value = float(input)
        return value >= min_val
    except ValueError:
        return False

def get_user_input():
    print("\n\033[1mPlease enter customer details:\033[0m")
    
    # Gender with validation
    gender = get_valid_input(
        "Gender (Male/Female): ",
        lambda x: x.lower() in ['male', 'female', 'm', 'f'],
        "Please enter Male or Female"
    ).lower()[0]
    gender = 1 if gender == 'f' else 0

    # Senior Citizen
    senior = get_valid_input(
        "Senior Citizen? (Yes/No): ",
        validate_yes_no,
        "Please enter Yes or No"
    ).lower()[0]
    senior_citizen = 1 if senior == 'y' else 0

    # Yes/No questions helper
    def get_yn(prompt):
        return get_valid_input(
            prompt,
            validate_yes_no,
            "Please enter Yes or No"
        ).lower()[0] == 'y'

    partner = get_yn("Has partner? (Yes/No): ")
    dependents = get_yn("Has dependents? (Yes/No): ")
    phone_service = get_yn("Has phone service? (Yes/No): ")
    multiple_lines = get_yn("Has multiple lines? (Yes/No): ")
    online_security = get_yn("Has online security? (Yes/No): ")
    online_backup = get_yn("Has online backup? (Yes/No): ")
    device_protection = get_yn("Has device protection? (Yes/No): ")
    tech_support = get_yn("Has tech support? (Yes/No): ")
    streaming_tv = get_yn("Has streaming TV? (Yes/No): ")
    streaming_movies = get_yn("Has streaming movies? (Yes/No): ")
    paperless_billing = get_yn("Uses paperless billing? (Yes/No): ")

    # Internet Service
    internet_service = get_valid_input(
        "Internet Service (DSL/Fiber optic/No): ",
        validate_internet,
        "Please enter DSL, Fiber optic, or No"
    ).lower()

    # Contract
    contract = get_valid_input(
        "Contract type (Month-to-month/One year/Two year): ",
        validate_contract,
        "Please enter Month-to-month, One year, or Two year"
    ).lower()

    # Payment Method
    payment_method = get_valid_input(
        "Payment method [Bank transfer (automatic)/Credit card (automatic)/Electronic check/Mailed check]: ",
        validate_payment,
        "Please enter one of the specified payment methods"
    ).lower()

    # Numerical inputs helper
    def get_num(prompt, min_val=0):
        return float(get_valid_input(
            prompt,
            lambda x: validate_number(x, min_val),
            f"Please enter a number greater than or equal to {min_val}"
        ))

    tenure = get_num("Tenure (months): ")
    monthly_charges = get_num("Monthly charges ($): ")
    total_charges = get_num("Total charges ($): ")

    # Build feature dictionary
    features = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': int(partner),
        'Dependents': int(dependents),
        'tenure': tenure,
        'PhoneService': int(phone_service),
        'MultipleLines': int(multiple_lines),
        'OnlineSecurity': int(online_security),
        'OnlineBackup': int(online_backup),
        'DeviceProtection': int(device_protection),
        'TechSupport': int(tech_support),
        'StreamingTV': int(streaming_tv),
        'StreamingMovies': int(streaming_movies),
        'PaperlessBilling': int(paperless_billing),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService_DSL': 1 if internet_service == 'dsl' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'no' else 0,
        'Contract_Month-to-month': 1 if contract == 'month-to-month' else 0,
        'Contract_One year': 1 if contract == 'one year' else 0,
        'Contract_Two year': 1 if contract == 'two year' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if 'bank transfer' in payment_method else 0,
        'PaymentMethod_Credit card (automatic)': 1 if 'credit card' in payment_method else 0,
        'PaymentMethod_Electronic check': 1 if 'electronic check' in payment_method else 0,
        'PaymentMethod_Mailed check': 1 if 'mailed check' in payment_method else 0,
    }

    # Create and scale DataFrame
    df_input = pd.DataFrame([features])
    df_input[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df_input[['tenure', 'MonthlyCharges', 'TotalCharges']])
    return df_input[feature_names].values

# Get prediction
user_features = get_user_input()
prediction_prob = model.predict(user_features)[0][0]
result = "Churn" if prediction_prob > 0.5 else "No Churn"

print(f"\n\033[1mPrediction Result:\033[0m")
print(f"Probability: {prediction_prob:.2%}")
print(f"Conclusion: {result}")