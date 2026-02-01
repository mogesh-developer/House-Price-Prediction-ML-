from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import numpy as np
import traceback

app = FastAPI()

# --- NEW: MUST ADD CORS TO CONNECT TO FRONTEND ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all websites to connect
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# Load files
try:
    with open('models/house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/LabelEncoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print("✅ Step 2 Complete: Model and Encoder loaded successfully")
except Exception as e:
    print(f"❌ ERROR LOADING FILES: {e}")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict_price(data: dict):
    try:
        # 1. Encode location
        loc_name = data.get("location")
        loc_encoded = le.transform([loc_name])[0]
        
        # 2. Build feature array
        features = np.array([[
            float(data["area"]),
            float(data["bedrooms"]),
            float(data["bathrooms"]),
            float(loc_encoded)
        ]])
        
        # 3. Predict
        prediction = model.predict(features)
        return {"predicted_price": float(prediction[0])}
        
    except Exception as e:
        # This will print the error to your VS Code terminal
        print("--- API CRASHED ---")
        traceback.print_exc() 
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)