# anamoly-detection
# 🧠 Anomaly Detection Using Machine Learning

This project demonstrates how to detect anomalies in highly imbalanced datasets using techniques like Isolation Forest and SMOTE.

## 🔧 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Imbalanced-learn (SMOTE)

## 🚀 How to Run

```bash
pip install -r requirements.txt
python app.py

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open(r'C:\Users\bharg\Downloads\Projectnest-main\Projectnest-main\anomaly project (4)\anomaly project\anomaly project\model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/About.html')
def about():
    return render_template('About.html')

@app.route('/predict.html')
def predict_form():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def predict():
    type = request.form['type']
    amount = float(request.form['amount'])
    old_balance = float(request.form['old_balance'])
    new_balance = float(request.form['new_balance'])

    prediction = model.predict([[type, amount, old_balance, new_balance]])

    return render_template('submit.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
