import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Month': [1, 2, 3, 4, 5, 6],
    'Sales': [100, 120, 140, 160, 180, 200]
}

df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

new_month = pd.DataFrame(
    [[7]],
    columns=['Month']
)

prediction = model.predict(new_month)

print("Predicted Future Sales:", prediction[0])
