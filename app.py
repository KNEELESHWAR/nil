import pickle
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__,template_folder='templates')

# Dictionary to map model names to filenames
model_files = {
    "SVC": "SVC.pkl",
    "Logistic Regression": "LogisticRegression.pkl",
    "Random Forest": "RandomForestClassifier.pkl",
    "K-Nearest Neighbors": "KNeighborsClassifier.pkl"
}

# Load the model
def load_model(model_name):
    with open(model_files[model_name], "rb") as model_file:
        model = pickle.load(model_file)
    return model

# Prediction function
def predict_loan_status(model_choice, gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                        credit_history, property_area):
    # Load selected model
    model = load_model(model_choice)

    # Create DataFrame for input
    user_input = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })

    prediction = model.predict(user_input)
    return "Approved" if prediction[0] == 1 else "Not Approved"

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    model_choice = request.form['model_choice']
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    dependents = int(request.form['dependents'])
    education = int(request.form['education'])
    self_employed = int(request.form['self_employed'])
    applicant_income = float(request.form['applicant_income'])
    coapplicant_income = float(request.form['coapplicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_amount_term = int(request.form['loan_amount_term'])
    credit_history = int(request.form['credit_history'])
    property_area = int(request.form['property_area'])

    # Get prediction
    prediction = predict_loan_status(model_choice, gender, married, dependents, education, self_employed,
                                      applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                                      credit_history, property_area)

    return render_template('index.html', prediction_text=f"Loan Status: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
