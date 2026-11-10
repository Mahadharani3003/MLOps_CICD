import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create dataset
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

y = np.array([0, 0, 1, 1])

# Train model
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# Test prediction
prediction = model.predict([[2, 3]])

# Verify model
assert len(prediction) == 1

print("ML Model Test Passed!")
print("Prediction:", prediction[0])