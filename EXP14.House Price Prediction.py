import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Area': [500, 700, 900, 1100, 1300, 1500],
    'Bedrooms': [1, 2, 2, 3, 3, 4],
    'Price': [2000000, 3000000, 3500000, 4500000, 5500000, 6500000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

new_house = pd.DataFrame(
    [[1000, 3]],
    columns=['Area', 'Bedrooms']
)

prediction = model.predict(new_house)

print("Predicted House Price:", prediction[0])
