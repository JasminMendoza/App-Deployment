import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
s = pd.read_csv("social_media_usage.csv")

# Clean the data
def clean_sm(df, x):
    x = df[x]
    x = np.where(x == 1, 1, 0)
    return x

ss = pd.DataFrame({
    "sm_li" : clean_sm(s, "web1h"),
    "income" : np.where(s["income"] > 9, np.nan, s["income"]),
    "education" : np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": clean_sm(s, "par"),
    "married": clean_sm(s, "marital"),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
})
ss = ss.dropna()

# Features and target
X = ss[["age", "income", "female", "education", "parent", "married"]]
y = ss[["sm_li"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train logistic regression model
logreg = LogisticRegression(class_weight="balanced", random_state=42)
logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)

# Streamlit App Interface
st.title('LinkedIn Usage Prediction App')

# User Input for Prediction
st.write('### Predict LinkedIn Usage for New Data')

# Age input remains the same
age = st.number_input('Age', min_value=18, max_value=100, value=25)

# Education selectbox
education = st.selectbox(
    "Education level", 
    options=["Less than High school", "High school incomplete", "High School graduate", 
             "Some college, no degree", "Associates degree", "Bachelor's degree", 
             "Some postgrad", "Postgrad"]
)

# Map education levels to numeric values
education_map = {
    "Less than High school": 1,
    "High school incomplete": 2,
    "High School graduate": 3,
    "Some college, no degree": 4,
    "Associates degree": 5,
    "Bachelor's degree": 6,
    "Some postgrad": 7,
    "Postgrad": 8
}

education_numeric = education_map[education]

# Income selectbox
income = st.selectbox(
    "Income [in thousands]", 
    options=["<$10", "$10 to under $20", "$20 to under $30", "$30 to under $40", 
             "$40 to under $50", "$50 to under $75", "$75 to under $100", 
             "$100 to under $150", "$150 or higher"]
)

# Map income levels to numeric values in thousands
income_map = {
    "<$10": 1,
    "$10 to under $20": 2,
    "$20 to under $30": 3,
    "$30 to under $40": 4,
    "$40 to under $50": 5,
    "$50 to under $75": 6,
    "$75 to under $100": 7,
    "$100 to under $150": 8,
    "$150 or higher": 9
}

income_numeric = income_map[income]

# Gender selection
female = st.selectbox('Gender', ['Male', 'Female'])

# Parent status selection
parent = st.selectbox('Parent Status', ['Yes', 'No'])

# Marital status selection
married = st.selectbox('Marital Status', ['Single', 'Married'])

# Prepare user input for prediction
input_data = np.array([[age, income_numeric, 1 if female == 'Female' else 0, education_numeric, 
                        1 if parent == 'Yes' else 0, 1 if married == 'Married' else 0]])

# Predict
prediction = logreg.predict(input_data)
probability = logreg.predict_proba(input_data)[0][1]

# Display result
if prediction == 1:
    st.write(f'Predicted: LinkedIn User')
else:
    st.write(f'Predicted: Non-LinkedIn User')

st.write(f'Probability: {probability:.2f}')


