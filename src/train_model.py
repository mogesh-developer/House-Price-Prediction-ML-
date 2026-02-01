import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

os.makedirs('models', exist_ok=True)
df = pd.read_csv("data/house_data.csv")

# CORRECT: Use the encoder object
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

X = df[['area', 'bedrooms', 'bathrooms', 'location']] # Explicit column order
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Save both
with open("models/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/LabelEncoder.pkl", "wb") as f:
    pickle.dump(le, f) # Save the WHOLE object 'le'

print("✅ Step 1 Complete: Model and Encoder saved.")