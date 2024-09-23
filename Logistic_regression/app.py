import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_path = "logistic_regression_model.pkl"  
model = joblib.load(model_path)
scaler_path = 'scaler.pkl'
scaler = joblib.load(scaler_path)



# Define the Streamlit app
st.title('Titanic Survival Prediction')
st.write('Enter the details of the passenger to predict their survival.')

# Collect user inputs
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.radio('Sex', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, value=0)
sibsp = st.number_input('Number of Siblings/Spouses Aboard (SibSp)', min_value=0, value=0)
parch = st.number_input('Number of Parents/Children Aboard (Parch)', min_value=0, value=0)
fare = st.number_input('Fare', min_value=0.0, value=0.0)
embarked = st.selectbox('Port of Embarkation (Embarked)', ['C', 'Q', 'S'])

# Encode categorical variables
sex_encoded = 0 if sex == 'Female' else 1
embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Prepare the input data for prediction
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
st.write(f'Input data for prediction: {input_data}')


# Scale the input data
input_data_scaled = scaler.transform(input_data)
st.write(f'Scaled input data: {input_data_scaled}')

# Make predictions
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    survival_probability = model.predict_proba(input_data_scaled)[0][1]  # Probability of survival
    survival_status = 'Survived' if prediction[0] == 1 else 'Did not survive'
    st.write(f'The passenger is predicted to: {survival_status}')
    st.write(f'Survival Probability: {survival_probability:.2f}')

# Display additional information
st.write('**Note:** The model was trained on historical data and predictions may not be accurate for all cases.')
