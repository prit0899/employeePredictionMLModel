import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
# Train model

			
df = pd.read_csv('HR_comma_sep.csv')
x = df[['last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']]
left = df['left']

X_train, X_test, y_train, y_test = train_test_split(x, left, train_size=0.8, random_state=10)
lg = LogisticRegression()
lg.fit(X_train, y_train)
joblib.dump(lg, "attrition_model.joblib")
print("Model saved!")
