import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = {
    'Income': [20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000],
    'Credit_Score': [500, 550, 600, 650, 700, 750, 800, 850],
    'Loan_Approved': [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Income', 'Credit_Score']]
y = df['Loan_Approved']

model = GaussianNB()
model.fit(X, y)

new_customer = pd.DataFrame(
    [[55000, 720]],
    columns=['Income', 'Credit_Score']
)

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")
