from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
#import uvicorn

model = pickle.load(open('randomforest_modelY .pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Model working"

@app.route('/predict', methods=['POST'])
def predict():
    Age = request.form.get('Age')
    JobLevel = request.form.get('JobLevel')
    OverTime = request.form.get('OverTime')
    Shift = request.form.get('Shift')
    TotalWorkingYears = request.form.get('TotalWorkingYears')
    YearsAtCompany = request.form.get('YearsAtCompany')
    YearsInCurrentRole = request.form.get('YearsInCurrentRole')
    YearsWithCurrManager = request.form.get('YearsWithCurrManager')

    input_query = np.array([[
        Age, JobLevel, OverTime, Shift, TotalWorkingYears,
        YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager
    ]])


    result = model.predict(input_query)[0]

    return jsonify({'Attrition': str(result)})

#
# if __name__ == '__main__':
#     app.run(debug=True)
