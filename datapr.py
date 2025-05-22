import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras

# step1
df = pd.read_csv('customer_churn.csv')
# print(df.sample(5))


# step2
#since the customer id columnn was not useful we have drop that column it wasnt in use in machine learning
df.drop('customerID', axis= 'columns',inplace= True)
#df.sample(5) -- customer id column removed
# print(df.sample(5))
# print(df.dtypes) #shows the data-type of the columns

# step3
#since the datatype of total is in object format but it should be in float format as the other number consisting column so first lets see the values in total charges
print(df.TotalCharges.values) #--values are in string format lets change those values
print(df.MonthlyCharges.values) #its in number format

# step4
# print(pd.to_numeric(df.TotalCharges))  #Unable to parse string " " at position 488 which means there a null values also
print(pd.to_numeric(df.TotalCharges, errors='coerce')) #ignore the errors basically and convert the values which  are convertable

# step5
print(pd.to_numeric(df.TotalCharges, errors='coerce').isnull()) #this will tell you the values is null or not in booleams

#step6
print(df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()]) #this will print the table which contains null columns 

# step7
print(df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()].shape) #shows the number of the columns by using shape
print("Old data frame length")
print(df.shape) #shows output in (rows,columns)

#step8
# creating a new data frame and storing the values and droping dwn the null values
df = df[df.TotalCharges != ' ']
df.TotalCharges = pd.to_numeric(df.TotalCharges)
df1 = df[df.TotalCharges != ' ']
df.loc[df.TotalCharges != ' ', 'TotalCharges'] = pd.to_numeric(df1['TotalCharges'])
print("New data frame length")
print(df1.shape)


#step9 
# again checking the datatypes of the ccolumn in new dataframe
print(df1.dtypes) 

#step10
#changing the total charge column to float
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
#checking datatypes of new dataframe again
print(df1.dtypes)

# # Step 11: Tenure by Churn Status
# tenure_churn_no = df1[df1.Churn == 'No'].tenure
# tenure_churn_yes = df1[df1.Churn == 'Yes'].tenure

# # Step 12: Monthly Charges by Churn Status
# monthly_churn_no = df1[df1.Churn == 'No'].MonthlyCharges
# monthly_churn_yes = df1[df1.Churn == 'Yes'].MonthlyCharges

# # Create subplots
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # Plot 1: Tenure
# axs[0].hist([tenure_churn_yes, tenure_churn_no], color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
# axs[0].set_xlabel("Tenure")
# axs[0].set_ylabel("Number of Customers")
# axs[0].set_title("Customer Tenure by Churn Status")
# axs[0].legend()

# # Plot 2: Monthly Charges
# axs[1].hist([monthly_churn_yes, monthly_churn_no], color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
# axs[1].set_xlabel("Monthly Charges")
# axs[1].set_ylabel("Number of Customers")
# axs[1].set_title("Monthly Charges by Churn Status")
# axs[1].legend()

# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()


#step14
# now creating an func which will show the unique values of the column consist of datatype object 
print("\n unique values of the column consist of datatype object")
def ucv(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            print(f'{column}: {df[column].unique()}')
ucv(df1)

def cv(df):
    for column in df.columns:
            print(f'{column}: {df[column].unique()}')

#step15
#now just cleaning the values and just making it in yes and no ... in simple yes and no columns
df1.replace('No internet service', 'No', inplace= True)
df1.replace('No phone service', 'No', inplace= True)
ucv(df1)

# step16
#replacing yes and no with 0 and 1
yn_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in yn_columns:
    df1.loc[:, col] = df1[col].replace({'Yes': 1, 'No': 0})
print("\n after replacing yes and No")
cv(df1)


# step17
# replacing females and males values in 1 and 0 respectively so that it will be easy for training the model as the computer understands in number formart
df1['gender'] = df1['gender'].replace({'Female': 1, 'Male': 0})
# print(df1['gender'].unique()) #---> was checking the values


# step18
#hot-enconding of the categorical columns
df1 = pd.get_dummies(data=df1, columns=['InternetService', 'Contract', 'PaymentMethod'])
cv(df1)
print(df1.dtypes)

#step19
#scalling 
colscale = ['tenure', 'MonthlyCharges', 'TotalCharges']
df1[colscale] = scaler.fit_transform(df1[colscale])
print(df1.sample(5))
cv(df1)

#step20 
#lets split the data into x and y
x = df1.drop('Churn', axis='columns')
y = df1['Churn']


# step 21
# train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=5)
print(x_train.shape) #--checked the quantity of train which has been splitted
print(x_test.shape)

#step 22
# creating model
model = keras. Sequential([
    keras.layers.Dense(26,input_shape=(26,), activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train, epochs = 500)
# Save the model after training
model.save('Churnm001.h5')
print("Model saved successfully!")