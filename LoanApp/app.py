import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = 'Approved' if prediction[0] == 1 else 'Rejected'

    return render_template('index.html', prediction_text='Loan Status: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)