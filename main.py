import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Function to train the model
def train_model():
    df = pd.read_csv('loan.csv')

    # Fill missing values
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Label encode categorical columns
    label_encoders = {}
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    x = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    loan_id = request.form['Loan_ID']
    gender = request.form['Gender']
    married = request.form['Married']
    dependents = request.form['Dependents']
    education = request.form['Education']
    self_employed = request.form['Self_Employed']
    applicant_income = float(request.form['ApplicantIncome'])
    coapplicant_income = float(request.form['CoapplicantIncome'])
    loan_amount = float(request.form['LoanAmount'])
    loan_amount_term = float(request.form['Loan_Amount_Term'])
    credit_history = float(request.form['Credit_History'])
    property_area = request.form['Property_Area']

    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    for column in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        input_data[column] = label_encoders[column].transform(input_data[column].astype(str))

    prediction = model.predict(input_data)[0]
    loan_status = 'Approved' if prediction == 1 else 'Rejected'

    return render_template('result.html', loan_status=loan_status, loan_id=loan_id)


if __name__ == "__main__":
    train_model()
    app.run(debug=True)
