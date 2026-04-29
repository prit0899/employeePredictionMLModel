# 🧠 Employee Attrition Predictor

A machine learning system that predicts whether an employee is likely to leave an organization — deployed via **FastAPI** and consumed by an **iOS mobile app**.

---

## 🚨 Problem Statement

HR teams struggle to identify employees at risk of leaving before it's too late. Replacing an employee costs 50–200% of their annual salary. Early prediction allows proactive retention strategies.

---

## 💡 Solution

A Logistic Regression model trained on 15,000 real HR records that predicts attrition risk with **75% accuracy** and **76% recall** for at-risk employees.

---

## 🏗️ Architecture

```
iOS App  →  FastAPI Endpoint  →  Logistic Regression Model  →  Prediction Response
```

---

## 📊 Dataset

- **Source:** Kaggle HR Dataset (`HR_comma_sep.csv`)
- **Size:** 15,000 employee records
- **Target:** `left` (0 = Stayed, 1 = Left)

| Feature | Description |
|---|---|
| `last_evaluation` | Last performance evaluation score (0–1) |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Average monthly working hours |
| `time_spend_company` | Years at company |

---

## ⚠️ Key Challenge — Class Imbalance

Initial model predicted **zero** employees would leave (0% recall on class 1) because 75% of data was majority class (stayed).

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
[[1686  580]   → Stayed: 1686 correct
 [ 175  559]]  → Left: 559 correctly identified ✅
```

- **Accuracy:** 75%
- **Recall (at-risk employees):** 76%
- **Model:** Logistic Regression with `class_weight='balanced'`
- **Train/Test Split:** 80/20 on 15,000 records

---

## 🚀 API — FastAPI

### Run Locally

```bash
pip install fastapi uvicorn scikit-learn pandas joblib
uvicorn main:app --reload
```

### Predict Endpoint

```
GET http://localhost:8000/predict
```

**Parameters:**

| Param | Type | Example |
|---|---|---|
| `last_evaluation` | float | 0.8 |
| `number_project` | int | 6 |
| `average_montly_hours` | float | 250 |
| `time_spend_company` | float | 4 |

**Response:**

```json
{
  "prediction": "High Risk",
  "probability": 0.83
}
```

### Interactive Docs

```
http://localhost:8000/docs
```

---

## 📱 iOS App

Swift iOS client that:
- Takes employee details as input
- Hits the FastAPI `/predict` endpoint
- Displays **High Risk 🔴** or **Low Risk 🟢** with probability

---

## 🗂️ Project Structure

```
employeePredictionMLModel/
├── train.py                  ← Train model and save with joblib
├── main.py                   ← FastAPI server
├── attrition_model.joblib    ← Saved trained model
├── HR_comma_sep.csv          ← Dataset (training only)
└── README.md
```

---

## 🔧 How to Retrain Model

```bash
python train.py
# Output: Model trained and saved as attrition_model.joblib
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (Logistic Regression) |
| API | FastAPI + Uvicorn |
| Model Serialization | Joblib |
| Data Processing | Pandas, NumPy |
| Mobile Client | Swift (iOS) |
| Dataset | Kaggle HR Dataset |

---

## 👨‍💻 Author

**Prit** — iOS Developer (6 years) expanding into ML and AI Product Engineering.

- Built end-to-end: from raw CSV → trained model → REST API → iOS client
- Identified and fixed class imbalance issue independently
- Interested in AI-powered mobile products

---

## 📌 What I Learned

- Logistic Regression for binary classification
- Handling imbalanced datasets with `class_weight`
- Model serialization with joblib
- Deploying ML models as REST APIs with FastAPI
- Consuming Python ML APIs from Swift/iOS
