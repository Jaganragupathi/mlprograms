import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'RAM': [2, 3, 4, 6, 8, 12],
    'Storage': [32, 64, 64, 128, 256, 256],
    'Price': [8000, 10000, 13000, 18000, 25000, 35000]
}

df = pd.DataFrame(data)

X = df[['RAM', 'Storage']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

new_mobile = pd.DataFrame(
    [[6, 128]],
    columns=['RAM', 'Storage']
)

prediction = model.predict(new_mobile)

print("Predicted Mobile Price:", prediction[0])
