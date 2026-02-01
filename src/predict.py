import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Create some sample training data
# Features: [area, bedrooms, bathrooms, location]
X = np.array([
    [1000, 2, 1, 1],
    [1200, 3, 2, 2], 
    [1500, 4, 2, 1],
    [800, 1, 1, 3],
    [2000, 4, 3, 1]
])

# Sample prices
y = np.array([200000, 280000, 350000, 150000, 450000])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
pickle.dump(model, open('house_price_model.pkl', 'wb'))
print("Test model created successfully!")