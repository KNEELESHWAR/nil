from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")
df = pd.DataFrame(data)

# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Encode categorical features
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
df['Credit_History'] = df['Credit_History'].astype(int)
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

# Define features and target
x = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(max_depth=4, random_state=10)
model.fit(x_train, y_train)

# Prediction function
def predict_loan_status(model, user_input, cibil_score):
    high_threshold = 750
    low_threshold = 650
    model_pred = model.predict(user_input)[0]
    
    if cibil_score >= high_threshold:
        return 'Yes'
    elif cibil_score < low_threshold:
        return 'No'
    else:
        return 'Yes' if model_pred == 1 else 'No'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        applicant_income = float(request.form['applicant_income'])
        loan_amount = float(request.form['loan_amount'])
        credit_history = int(request.form['credit_history'])
        cibil_score = int(request.form['cibil_score'])
        
        # Prepare input data for prediction
        user_input = np.array([[gender, married, applicant_income, loan_amount, credit_history]])
        
        # Get prediction
        loan_status = predict_loan_status(model, user_input, cibil_score)
        
        # Return result
        return render_template('index.html', prediction=loan_status)

if __name__ == '__main__':
    app.run(debug=True)
