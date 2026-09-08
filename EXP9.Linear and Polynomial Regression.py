import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [35, 40, 50, 65, 80, 90, 95, 97]
}

df = pd.DataFrame(data)

X = df[['Study_Hours']]
y = df['Marks']

linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

print("Linear Regression")
print("Predicted Values:", linear_pred)
print("R2 Score:", r2_score(y, linear_pred))
print("Mean Squared Error:", mean_squared_error(y, linear_pred))

print("\nPolynomial Regression")
print("Predicted Values:", poly_pred)
print("R2 Score:", r2_score(y, poly_pred))
print("Mean Squared Error:", mean_squared_error(y, poly_pred))
