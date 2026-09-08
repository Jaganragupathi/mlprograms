import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Age': [22, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Smoking_Years': [0, 2, 4, 6, 10, 15, 20, 25, 30, 35],
    'Oxygen_Level': [98, 97, 96, 95, 93, 91, 89, 87, 85, 83],
    'Lung_Disease': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Age', 'Smoking_Years', 'Oxygen_Level']]
y = df['Lung_Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values:", list(y_test))
print("Predicted Values:", list(y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
