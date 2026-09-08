import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Age': [1, 2, 3, 4, 5, 6],
    'Mileage': [10000, 20000, 30000, 40000, 50000, 60000],
    'Price': [900000, 800000, 700000, 600000, 500000, 400000]
}

df = pd.DataFrame(data)

X = df[['Age', 'Mileage']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

new_car = pd.DataFrame(
    [[3, 30000]],
    columns=['Age', 'Mileage']
)

prediction = model.predict(new_car)

print("Predicted Car Price:", prediction[0])
