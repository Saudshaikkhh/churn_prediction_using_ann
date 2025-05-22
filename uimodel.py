import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #f1f5f9;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 20px;
        background-color: #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }
    .bold-text {
        font-weight: 600;
    }
    .centered {
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #64748b;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        padding: 2rem 1rem;
    }
    /* Hide the Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return keras.models.load_model('Churnm001.h5')
    except:
        st.error("Error loading model file. Please make sure 'Churnm001.h5' is available.")
        st.stop()

@st.cache_data
def load_and_process_data():
    try:
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
        
        # Create and fit the scaler
        scaler = MinMaxScaler()
        scaler.fit(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        
        return df, scaler
    except:
        st.error("Error loading data. Please make sure 'customer_churn.csv' is available.")
        st.stop()

# Define feature names
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService_DSL',
                'InternetService_Fiber optic', 'InternetService_No',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract_Month-to-month',
                'Contract_One year', 'Contract_Two year', 'PaperlessBilling',
                'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                'MonthlyCharges', 'TotalCharges']

# Load model and data
model = load_model()
df, scaler = load_and_process_data()

# App header
st.markdown('<h1 class="main-header">Customer Churn Prediction System</h1>', unsafe_allow_html=True)

# Layout with tabs
tab1, tab2, tab3 = st.tabs(["Prediction Tool", "Data Insights", "About"])

with tab1:
    st.markdown('<p class="centered">Enter customer information to predict likelihood of churn</p>', unsafe_allow_html=True)
    
    # Create two columns for the form
    col1, col2 = st.columns(2)
    
    # Column 1 inputs
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Customer Demographics</p>', unsafe_allow_html=True)
        
        gender = st.radio("Gender", options=["Male", "Female"])
        senior_citizen = st.radio("Senior Citizen", options=["No", "Yes"])
        partner = st.radio("Has Partner", options=["No", "Yes"])
        dependents = st.radio("Has Dependents", options=["No", "Yes"])
        
        st.markdown('<p class="sub-header">Account Information</p>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        contract = st.selectbox("Contract Type", 
                             options=["Month-to-month", "One year", "Two year"])
        paperless_billing = st.radio("Paperless Billing", options=["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                   options=["Electronic check", "Mailed check", 
                                          "Bank transfer (automatic)", "Credit card (automatic)"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 2 inputs
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Services</p>', unsafe_allow_html=True)
        
        phone_service = st.radio("Phone Service", options=["No", "Yes"])
        multiple_lines = st.radio("Multiple Lines", options=["No", "Yes"], 
                               disabled=not (phone_service == "Yes"))
        
        internet_service = st.selectbox("Internet Service", 
                                     options=["DSL", "Fiber optic", "No"])
        
        internet_options_disabled = (internet_service == "No")
        
        online_security = st.radio("Online Security", options=["No", "Yes"], 
                                disabled=internet_options_disabled)
        online_backup = st.radio("Online Backup", options=["No", "Yes"], 
                              disabled=internet_options_disabled)
        device_protection = st.radio("Device Protection", options=["No", "Yes"], 
                                  disabled=internet_options_disabled)
        tech_support = st.radio("Tech Support", options=["No", "Yes"], 
                             disabled=internet_options_disabled)
        streaming_tv = st.radio("Streaming TV", options=["No", "Yes"], 
                             disabled=internet_options_disabled)
        streaming_movies = st.radio("Streaming Movies", options=["No", "Yes"], 
                                 disabled=internet_options_disabled)
        
        st.markdown('<p class="sub-header">Charges</p>', unsafe_allow_html=True)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=0.0, max_value=150.0, 
                                 value=65.0, step=5.0)
        
        # Calculate suggested total charges based on tenure and monthly charges
        suggested_total = monthly_charges * tenure
        total_charges = st.number_input("Total Charges ($)", 
                                     min_value=0.0, 
                                     value=float(suggested_total),
                                     step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("Predict Churn Probability", type="primary", use_container_width=True):
        with st.spinner("Processing prediction..."):
            # Prepare features based on user input
            features = {
                'gender': 1 if gender == "Female" else 0,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0,
                'Dependents': 1 if dependents == "Yes" else 0,
                'tenure': tenure,
                'PhoneService': 1 if phone_service == "Yes" else 0,
                'MultipleLines': 1 if multiple_lines == "Yes" and phone_service == "Yes" else 0,
                'OnlineSecurity': 1 if online_security == "Yes" and internet_service != "No" else 0,
                'OnlineBackup': 1 if online_backup == "Yes" and internet_service != "No" else 0,
                'DeviceProtection': 1 if device_protection == "Yes" and internet_service != "No" else 0,
                'TechSupport': 1 if tech_support == "Yes" and internet_service != "No" else 0,
                'StreamingTV': 1 if streaming_tv == "Yes" and internet_service != "No" else 0,
                'StreamingMovies': 1 if streaming_movies == "Yes" and internet_service != "No" else 0,
                'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'InternetService_DSL': 1 if internet_service == "DSL" else 0,
                'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
                'InternetService_No': 1 if internet_service == "No" else 0,
                'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
                'Contract_One year': 1 if contract == "One year" else 0,
                'Contract_Two year': 1 if contract == "Two year" else 0,
                'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
                'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
                'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
                'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
            }
            
            # Create dataframe and scale
            df_input = pd.DataFrame([features])
            df_input[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
                df_input[['tenure', 'MonthlyCharges', 'TotalCharges']])
                
            # Get prediction
            time.sleep(1)  # Small delay for UX
            prediction_prob = model.predict(df_input[feature_names].values)[0][0]
            churn_status = "High Risk of Churn" if prediction_prob > 0.5 else "Low Risk of Churn"
        
        # Display prediction results in an eye-catching way
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="centered">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Create columns for the gauge and text
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            # Create a gauge chart with Plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(prediction_prob * 100),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'green'},
                        {'range': [30, 70], 'color': 'yellow'},
                        {'range': [70, 100], 'color': 'red'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50}}))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with res_col2:
            st.markdown(f"""
            <div style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <h3 style="margin-bottom: 20px;">Customer Assessment</h3>
                <p style="font-size: 24px; font-weight: 600; color: {'red' if prediction_prob > 0.5 else 'green'};">
                    {churn_status}
                </p>
                <p style="font-size: 18px;">Churn Probability: {prediction_prob:.1%}</p>
                <p style="margin-top: 30px;">
                    {'This customer is at high risk of churning. Consider retention strategies.' 
                     if prediction_prob > 0.5 else 
                     'This customer is likely to stay. Consider growth opportunities.'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk factors
        if prediction_prob > 0.3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3>Key Risk Factors</h3>', unsafe_allow_html=True)
            
            risk_factors = []
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract (higher churn risk compared to longer-term contracts)")
            if internet_service == "Fiber optic":
                risk_factors.append("Fiber optic internet service (historically associated with higher churn)")
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment method (correlates with higher churn)")
            if tenure < 12:
                risk_factors.append(f"Short tenure ({tenure} months)")
            if monthly_charges > 80:
                risk_factors.append(f"High monthly charges (${monthly_charges:.2f})")
            if tech_support == "No" and internet_service != "No":
                risk_factors.append("No tech support subscription")
            if online_security == "No" and internet_service != "No":
                risk_factors.append("No online security subscription")
                
            if not risk_factors:
                risk_factors.append("Other factors are contributing to the churn risk")
                
            for factor in risk_factors:
                st.markdown(f"- {factor}")
                
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3>Retention Recommendations</h3>', unsafe_allow_html=True)
            
            recommendations = []
            if contract == "Month-to-month":
                recommendations.append("Offer incentives for longer-term contracts")
            if tech_support == "No" and internet_service != "No":
                recommendations.append("Promote tech support benefits")
            if online_security == "No" and internet_service != "No":
                recommendations.append("Highlight online security features")
            if monthly_charges > 80:
                recommendations.append("Review pricing structure or offer targeted discounts")
            if tenure < 12:
                recommendations.append("Implement early engagement program to increase loyalty")
                
            if not recommendations:
                recommendations.append("Personalized retention strategy based on usage patterns")
                
            for rec in recommendations:
                st.markdown(f"- {rec}")
                
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 class="sub-header">Data Insights & Visualization</h2>', unsafe_allow_html=True)
    
    # Information about the dataset
    st.markdown('<div class="card">', unsafe_allow_html=True)
    total_customers = len(df)
    churn_rate = df['Churn'].mean() * 100
    
    st.markdown(f"""
    ### Dataset Overview
    - **Total customers**: {total_customers}
    - **Overall churn rate**: {churn_rate:.1f}%
    - **Features available**: {len(df.columns)}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Churn Rate by Contract Type")
        
        # Calculate churn rate by contract type
        contract_churn = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn Rate': [
                df[df['Contract_Month-to-month'] == 1]['Churn'].mean() * 100,
                df[df['Contract_One year'] == 1]['Churn'].mean() * 100,
                df[df['Contract_Two year'] == 1]['Churn'].mean() * 100
            ]
        })
        
        # Create bar chart
        fig = px.bar(
            contract_churn, 
            x='Contract', 
            y='Churn Rate',
            color='Churn Rate',
            color_continuous_scale=['green', 'yellow', 'red'],
            labels={'Churn Rate': 'Churn Rate (%)'},
            text_auto='.1f'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Tenure vs. Churn")
        
        # Group by tenure ranges
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                  labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
        tenure_churn = df.groupby('tenure_group')['Churn'].mean().reset_index()
        tenure_churn['Churn Rate'] = tenure_churn['Churn'] * 100
        
        # Create line chart
        fig = px.line(
            tenure_churn, 
            x='tenure_group', 
            y='Churn Rate',
            markers=True,
            labels={'tenure_group': 'Tenure (months)', 'Churn Rate': 'Churn Rate (%)'},
        )
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with viz_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Internet Service Impact")
        
        # Get internet service types
        internet_churn = pd.DataFrame({
            'Internet Service': ['DSL', 'Fiber optic', 'No Internet'],
            'Churn Rate': [
                df[df['InternetService_DSL'] == 1]['Churn'].mean() * 100,
                df[df['InternetService_Fiber optic'] == 1]['Churn'].mean() * 100,
                df[df['InternetService_No'] == 1]['Churn'].mean() * 100
            ]
        })
        
        # Create bar chart
        fig = px.bar(
            internet_churn, 
            x='Internet Service', 
            y='Churn Rate',
            color='Churn Rate',
            color_continuous_scale=['green', 'yellow', 'red'],
            labels={'Churn Rate': 'Churn Rate (%)'},
            text_auto='.1f'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Additional Services Impact")
        
        # Calculate churn rate for various services
        services_impact = pd.DataFrame({
            'Service': ['Online Security', 'Tech Support', 'Device Protection', 'Online Backup'],
            'With Service': [
                df[df['OnlineSecurity'] == 1]['Churn'].mean() * 100,
                df[df['TechSupport'] == 1]['Churn'].mean() * 100,
                df[df['DeviceProtection'] == 1]['Churn'].mean() * 100,
                df[df['OnlineBackup'] == 1]['Churn'].mean() * 100
            ],
            'Without Service': [
                df[(df['OnlineSecurity'] == 0) & (df['InternetService_No'] == 0)]['Churn'].mean() * 100,
                df[(df['TechSupport'] == 0) & (df['InternetService_No'] == 0)]['Churn'].mean() * 100,
                df[(df['DeviceProtection'] == 0) & (df['InternetService_No'] == 0)]['Churn'].mean() * 100,
                df[(df['OnlineBackup'] == 0) & (df['InternetService_No'] == 0)]['Churn'].mean() * 100
            ]
        })
        
        # Reshape for grouped bar chart
        services_impact_melted = pd.melt(
            services_impact, 
            id_vars=['Service'], 
            value_vars=['With Service', 'Without Service'],
            var_name='Status', 
            value_name='Churn Rate'
        )
        
        # Create grouped bar chart
        fig = px.bar(
            services_impact_melted, 
            x='Service', 
            y='Churn Rate',
            color='Status',
            barmode='group',
            labels={'Churn Rate': 'Churn Rate (%)'},
            color_discrete_map={'With Service': '#3B82F6', 'Without Service': '#EF4444'},
            text_auto='.1f'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    # About This Application
    
    This Customer Churn Prediction application uses machine learning to predict the likelihood of customer churn based on various customer attributes and service usage patterns.
    
    ## What is Customer Churn?
    
    Customer churn refers to when customers stop doing business with a company or service. In the telecommunications industry, this typically means customers who discontinue their subscription to services.
    
    ## How It Works
    
    1. The application uses a pre-trained TensorFlow neural network model to make predictions
    2. The model was trained on historical customer data with churn outcomes
    3. Features like contract type, payment method, services used, and tenure are analyzed
    4. The model outputs a probability score indicating the likelihood of churn
    
    ## Model Information
    
    - **Model Type**: Neural Network (TensorFlow/Keras)
    - **Features Used**: 26 customer and service attributes
    - **Output**: Probability of customer churn (0-100%)
    
    ## How to Use
    
    1. Enter customer information in the "Prediction Tool" tab
    2. Click "Predict Churn Probability" to see results
    3. Review the risk factors and recommendations if provided
    4. Explore data insights in the "Data Insights" tab
    
    ## Data Privacy
    
    This application runs locally and does not store any entered customer information.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Customer Churn Prediction System | Created with Streamlit | © 2025
</div>
""", unsafe_allow_html=True)