# 🧠 Employee Attrition Predictor

A machine learning system that predicts whether an employee is likely to leave an organization — deployed live via **FastAPI on Render** and consumed by an **iOS mobile app**.

🔴 **Live API:** https://employeepredictionmlmodel.onrender.com

📱 **iOS App Repo:** [employeePrediction](https://github.com/prit0899/employeePrediction)

---

## 🚨 Problem Statement

HR teams struggle to identify employees at risk of leaving before it's too late. Replacing an employee costs 50–200% of their annual salary. Early prediction allows proactive retention strategies.

---

## 💡 Solution

A Logistic Regression model trained on 15,000 real HR records that predicts attrition risk with **75% accuracy** and **76% recall** for at-risk employees — served via a REST API and consumed from an iOS app.

---

## 🏗️ Architecture

```
iOS App (Swift)
     ↓  HTTP Request
FastAPI Server (Render Cloud)
     ↓  Loads saved model
Logistic Regression (scikit-learn)
     ↓  Returns JSON
iOS App displays result
```

---

## 🌐 Live API Usage

**Base URL:**
```
https://employeepredictionmlmodel.onrender.com
```

**Predict Endpoint:**
```
GET /predict
```

**Example Request:**
```
https://employeepredictionmlmodel.onrender.com/predict?last_evaluation=0.8&number_project=5&average_montly_hours=220&time_spend_company=3
```

**Response:**
```json
{
  "prediction": "High Risk",
  "probability": 0.83
}
```

**Interactive Docs:**
```
https://employeepredictionmlmodel.onrender.com/docs
```

> ⚠️ Free tier may sleep after inactivity. First request can take 20–30 seconds to wake up.

---

## 📊 Dataset

- **Source:** Kaggle HR Dataset (`HR_comma_sep.csv`)
- **Size:** 15,000 employee records
- **Split:** 80% train / 20% test
- **Target:** `left` (0 = Stayed, 1 = Left)

| Feature | Description |
|---|---|
| `last_evaluation` | Last performance evaluation score (0–1) |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Average monthly working hours |
| `time_spend_company` | Years at company |

---

## ⚠️ Key Challenge — Class Imbalance

Initial model predicted **zero** employees would leave (0% recall on class 1) because 75% of data was majority class.

**Fix:** Used `class_weight='balanced'` in Logistic Regression.

| Metric | Before Fix | After Fix |
|---|---|---|
| Recall (Left) | 0% | 76% |
| F1 Score (Left) | 0.00 | 0.60 |
| Accuracy | 74% | 75% |

---

## 📈 Model Performance

```
Confusion Matrix:
[[1686  580]   → Stayed: 1686 correctly identified
 [ 175  559]]  → Left:   559 correctly identified ✅
```

- **Accuracy:** 75%
- **Recall (at-risk employees):** 76%
- **Model:** Logistic Regression with `class_weight='balanced'`

---

## 🗂️ Project Structure

```
employeePredictionMLModel/
├── train.py                  ← Train model and save with joblib
├── main.py                   ← FastAPI server (deployed on Render)
├── attrition_model.joblib    ← Saved trained model
├── HR_comma_sep.csv          ← Dataset (used for training only)
├── requirements.txt          ← Python dependencies
└── README.md
```

---

## 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (generates attrition_model.joblib)
python train.py

# Start API server
uvicorn main:app --reload

# Test at
http://localhost:8000/docs
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (Logistic Regression) |
| API | FastAPI + Uvicorn |
| Model Serialization | Joblib |
| Data Processing | Pandas, NumPy |
| Deployment | Render.com |
| Mobile Client | Swift (iOS) |
| Dataset | Kaggle HR Dataset |

---

## 👨‍💻 Author

**Prit** — iOS Developer (6 years) expanding into ML and AI Product Engineering.

- Built end-to-end: raw CSV → trained model → REST API → cloud deployment → iOS client
- Identified and fixed class imbalance problem independently
- Targeting AI Product Engineer roles combining mobile + ML expertise

---

## 📌 What I Learned

- Logistic Regression for binary classification
- Identifying and fixing class imbalance with `class_weight='balanced'`
- Model serialization with joblib
- Building and deploying ML APIs with FastAPI on Render
- Consuming Python ML APIs from Swift/iOS using URLSession
