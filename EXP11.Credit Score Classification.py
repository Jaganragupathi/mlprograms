import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    'Income': [20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000],
    'Credit_History': [1, 1, 0, 1, 1, 1, 0, 1],
    'Credit_Score': [0, 0, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Income', 'Credit_History']]
y = df['Credit_Score']

model = LogisticRegression()
model.fit(X, y)

new_data = pd.DataFrame(
    [[55000, 1]],
    columns=['Income', 'Credit_History']
)

prediction = model.predict(new_data)

print("Credit Score Class:", prediction[0])
