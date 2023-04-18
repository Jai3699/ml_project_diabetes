from flask import Flask,render_template,request,app,Response
import pickle
import numpy as np
import pandas as pd



app = Flask(__name__)
scaler=pickle.load(open('/config/workspace/models/scaler.pkl','rb'))
logistic=pickle.load(open('/config/workspace/models/logisctic.pkl','rb'))


@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict_datapoint():
        
    if request.method=='POST':
        Pregnancies=int(request.form['Pregnancies'])
        Glucose=float(request.form['Glucose'])
        BloodPressure=float(request.form['BloodPressure'])
        SkinThickness=float(request.form['SkinThickness'])
        Insulin=float(request.form['Insulin'])
        BMI=float(request.form['BMI'])
        DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
        Age=float(request.form['Age'])
        scaled_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        outcome=logistic.predict(scaled_data)
        if outcome[0]==1:
            result='Diabetic'
            return render_template('single_prediction.html',result=result)
        else:
            result='Non-Diabetic'
            return render_template('single_prediction.html',result=result)  
        
    else:
        return render_template('home.html')    

if __name__=="__main__":
    app.run(host="0.0.0.0")
