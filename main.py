import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier

# Load and preprocess data
df = pd.read_csv('insurance_data.csv', on_bad_lines='skip')
df['insurance_type'] = df['insurance_type'].apply(lambda x: x.split(','))

# Feature and target separation
X = df.drop(columns=['insurance_type'])
y = df['insurance_type']

# Binarize the insurance_type
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['age', 'health_status'])
    ], remainder='passthrough')

# Create the model pipeline
model = make_pipeline(
    preprocessor,
    MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
)

# Train the model
model.fit(X_train, y_train)

# Streamlit app
st.title('Insurance Recommendation System')

# User inputs
age = st.number_input('Age', min_value=18, max_value=100, value=30)
income = st.number_input('Income', min_value=0, value=50000)
occupation = st.selectbox('Occupation', df['occupation'].unique())
health_status = st.selectbox('Health Status', df['health_status'].unique())
dependents = st.number_input('Number of Dependents', min_value=0, value=0)

# Prediction
user_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'occupation': [occupation],
    'health_status': [health_status],
    'dependents': [dependents]
})

prediction = model.predict(user_data)
recommended_insurance = mlb.inverse_transform(prediction)

st.write(f'Recommended Insurance Types: {", ".join(recommended_insurance[0])}')
