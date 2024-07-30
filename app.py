from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
with open('credit.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    Annual_Income = float(data['Annual Income'])
    Num_Bank_Accounts = float(data['Number of Bank Accounts'])
    Num_Credit_Card = float(data['Number of Credit Card'])
    Interest_Rate = float(data['Interest Rate'])
    Num_of_Loan = float(data['Number of Loan'])
    Delay_from_due_date = float(data['Delay from due date'])
    Num_of_Delayed_Payment = float(data['Number of Delayed Payment'])
    Changed_Credit_Limit = float(data['Changed Credit Limit'])
    Num_Credit_Inquiries = float(data['Number of Credit Inquires'])
    Outstanding_Debt = float(data['Outstanding Debt'])
    Total_EMI_per_month = float(data['Total EMI per Month'])
    Credit_Mix = float(data['Credit Mix'])

    # Calculate additional features
    Total_Num_Accounts = Num_Bank_Accounts + Num_Credit_Card
    Debt_Per_Account = Outstanding_Debt / Total_Num_Accounts
    Debt_to_Income_Ratio = Outstanding_Debt / Annual_Income
    Delayed_Payments_Per_Account = Num_of_Delayed_Payment / Total_Num_Accounts

    # Prepare feature array for prediction
    feature = np.array([[Annual_Income, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan,
                         Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit,
                         Num_Credit_Inquiries, Outstanding_Debt, Total_EMI_per_month,Credit_Mix,
                         Total_Num_Accounts, Debt_Per_Account, Debt_to_Income_Ratio, Delayed_Payments_Per_Account]])

    # Make prediction
    prediction = model.predict(feature)

    return render_template('result.html', pred_res=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
