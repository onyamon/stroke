from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# -------------------------
# โหลดโมเดล
# -------------------------
model = joblib.load("best_model.pkl")

@app.get("/")
def home():
    return {"message": "Stroke Prediction API Running"}

# -------------------------
# รับ input + ทำนาย
# -------------------------
@app.post("/predict")
def predict(data: dict):

    # รับ 5 ตัวจากผู้ใช้
    age = data["age"]
    hypertension = data["hypertension"]
    heart = data["heart_disease"]
    glucose = data["avg_glucose_level"]
    bmi = data["bmi"]

    # เติมค่า default ให้ครบ 10 features
    gender = 0
    married = 1
    work = 0
    residence = 1
    smoke = 0

    features = np.array([
        gender, age, hypertension, heart,
        married, work, residence,
        glucose, bmi, smoke
    ]).reshape(1,-1)

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}
