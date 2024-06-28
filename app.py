import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv




# Creating a DataFrame
df = pd.read_csv("customer_policy_data.csv", dtype=str)


# Encoding categorical variables
label_encoders = {}
for column in ["Customer ID","Customer Name","Mobile Number","Email Id", "Gender", "Marital Status", "Location", "Occupation", "Claim Type", "Online Activity", "Current Insurance", "Coverage", "Renewal"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Handle missing values in 'Premium (3yrs)'
#df["Premium (3yrs)"] = df["Premium (3yrs)"].str.replace('$', '').str.replace(',', '').replace('-', '0').astype(float)

# Split features and target
X = df.drop(["Policy Number","Recommended Insurance"], axis=1)
#X = df.drop([ "Recommended Insurance"], axis=1)
y = df["Recommended Insurance"]

# Encode target labels
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to predict recommended insurance types for new customers
def predict_recommended_insurance(new_customer_data):
    # Preprocess the new customer data
    for column, le in label_encoders.items():
        new_customer_data[column] = le.transform([new_customer_data[column]])[0]
    
    # Handle the premium value
    new_customer_data["Premium (3yrs)"] = float(new_customer_data["Premium (3yrs)"].replace('$', '').replace(',', ''))

    # Convert to DataFrame
    new_customer_df = pd.DataFrame([new_customer_data])
    
    # Scale the features
    new_customer_df = scaler.transform(new_customer_df)
    
    # Predict using the trained model
    predicted_label = model.predict(new_customer_df)
    recommended_insurance = target_encoder.inverse_transform(predicted_label)
   # print(recommended_insurance)
    return recommended_insurance[0]

def  getDataFromCSV(columnName,inputData):
  with open("customer_policy_data.csv") as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
        if (row[columnName] == inputData):
            return row     
def getCustomerData(row):
    new_customer_data = {
              "Customer ID": row['Customer ID'],
              "Customer Name": row['Customer Name'],
              "Mobile Number": row['Mobile Number'],
              "Email Id": row['Email Id'],
              "Age": row['Age'],
              "Gender": row['Gender'],
              "Marital Status": row['Marital Status'],
              "Dependents": row['Dependents'],
              "Location": row['Location'],
              "Occupation": row['Occupation'],
              "Claims (3yrs)": row['Claims (3yrs)'],
              "Claim Type": row['Claim Type'],
              "Online Activity": row['Online Activity'],
              "Service Calls": row['Service Calls'],
              "Current Insurance": row['Current Insurance'],
              "Coverage": row['Coverage'],
              "Renewal": row['Renewal'],
              "Premium (3yrs)": row['Premium (3yrs)']
               }
    return new_customer_data    

# Streamlit app
#st.image("VSoft-Logo.png", width=400)
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("VSoft-Logo.png")
st.header('Insurance Recommendation System')

st.markdown("""
<style>
.stTextInput > label {
font-size:120%;
font-weight:bold;
color:black;
background:linear-gradient(to bottom, #ffffff 0%,#ffffff 100%);
border: 2px;
border-radius: 3px;
}

[data-baseweb="base-input"]{
background:linear-gradient(to bottom, #e8ede8 0%,#e8ede8 100%);
border: 2px;
border-radius: 3px;
}

input[class]{
font-weight: normal;
font-size:120%;
color: black;
}
</style>
""", unsafe_allow_html=True)

# Input fields for policy number data
policy_number = st.text_input('Policy Number')

st.write("Or")

# Input fields for policy number data
mobile_number = st.text_input('Mobile Number')

st.write("Or")

# Input fields for policy number data
email_id = st.text_input('Email Id')

inputData = None
columnName = None
# Predict button
if st.button('Predict Recommended Insurance'):         
    if policy_number:
        inputData = policy_number
        columnName = 'Policy Number'
    elif mobile_number: 
        inputData = mobile_number
        columnName = 'Mobile Number'
    elif email_id:
        inputData = email_id
        columnName = 'Email Id'
    else:
        inputData = None
        columnName = None
    #Handle the inputData    
    if inputData:
        row = getDataFromCSV(columnName,inputData)
        recommended_insurance = predict_recommended_insurance(getCustomerData(row))
        st.subheader(f'Recommended Insurance: {recommended_insurance}')
        st.subheader(f'Customer Details:')
        dataframe=pd.DataFrame.from_dict(row,orient='index')
         # Display the dataframe with enhanced UI
        st.dataframe(dataframe)
      
    else:
      st.error("Error: No input provided. Please enter data in at least one input field.") 
                    


        
    
         
            

          


   