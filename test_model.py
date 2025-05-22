import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Load your data and preprocess exactly like before
df = pd.read_csv('customer_churn.csv')
df.drop('customerID', axis='columns', inplace=True)
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

scaler = MinMaxScaler()
colscale = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[colscale] = scaler.fit_transform(df[colscale])

x = df.drop('Churn', axis='columns')
y = df['Churn']

# Split data (you must split data the same way to get matching shapes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# Load the saved model
model = keras.models.load_model('Churnm001.h5')
print("Model loaded successfully!")

# Evaluate model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Make predictions on test data
predictions = model.predict(x_test)
# Since this is binary classification with sigmoid output, predictions are probabilities.
# To convert to binary class 0/1:
predicted_classes = (predictions > 0.5).astype(int)

print("Sample predictions:")
print(predicted_classes[:10].flatten())
