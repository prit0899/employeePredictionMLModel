import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
import joblib
# Train model

			
# Load once when server starts
lg = joblib.load("attrition_model.joblib")

# FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Attrition Predictor API"}

@app.get("/predict")
def predict(last_evaluation: float, number_project: int, average_montly_hours: float, time_spend_company: float):
    result = lg.predict([[float(last_evaluation), int(number_project), float(average_montly_hours), float(time_spend_company)]])
    probability = lg.predict_proba([[last_evaluation, number_project, average_montly_hours, time_spend_company]])
    
    return {
        "prediction":{
            "prediction": int(result[0]),#"High Risk" if int(result[0]) == 1 else "Low Risk"
            "probability": round(float(probability[0][1]), 2)
            }
        }
