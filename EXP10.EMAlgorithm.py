import numpy as np
from sklearn.mixture import GaussianMixture

X = np.array([
    [1], [2], [3], [4], [5],
    [10], [11], [12], [13], [14]
])

model = GaussianMixture(n_components=2, random_state=42)

model.fit(X)

labels = model.predict(X)

print("Data Points:", X.flatten())
print("Cluster Labels:", labels)
print("Means:", model.means_.flatten())
