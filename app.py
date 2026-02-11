import streamlit as st
import pandas as pd
import joblib

# =========================
# โหลดโมเดล
# =========================
model = joblib.load("best_model.pkl")

# =========================
# ตั้งค่าหน้าเว็บ
# =========================
st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="💖",
    layout="centered"
)

# =========================
# CSS ธีมหรู
# =========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom,#fff5f8,#ffe3ec);
}

h1,h2,h3 {
    text-align:center;
    color:#c2185b;
}

.stTextInput>div>div>input {
    border-radius:12px;
    padding:10px;
}

.stSelectbox>div>div {
    border-radius:12px;
}

.stButton>button {
    background: linear-gradient(45deg,#ff4da6,#ff80bf);
    color:white;
    font-size:18px;
    border-radius:12px;
    padding:12px 25px;
    border:none;
}

.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:20px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.title("💖 Stroke Prediction System")
st.write("ระบบประเมินความเสี่ยงโรคหลอดเลือดสมองเบื้องต้น")

st.divider()

# =========================
# INPUT UI
# =========================

age = st.text_input("อายุ (ปี)")

hypertension_text = st.selectbox(
    "ความดันโลหิตสูง",
    ["ไม่มี", "มี"]
)

heart_text = st.selectbox(
    "โรคหัวใจ",
    ["ไม่เป็น", "เป็น"]
)

glucose = st.text_input("ระดับน้ำตาลในเลือด")
bmi = st.text_input("BMI (ดัชนีมวลกาย)")

st.divider()

# =========================
# ปุ่มทำนาย
# =========================
if st.button("✨ ทำนายความเสี่ยง"):

    try:
        hypertension = 1 if hypertension_text == "มี" else 0
        heart = 1 if heart_text == "เป็น" else 0

        input_data = pd.DataFrame([{
            "age": float(age),
            "hypertension": hypertension,
            "heart_disease": heart,
            "avg_glucose_level": float(glucose),
            "bmi": float(bmi)
        }])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100

        st.divider()

        if pred == 1:
            st.markdown(
                f"<div class='result-box' style='background:#ffccd5;color:#b00020;'>"
                f"⚠️ มีความเสี่ยงโรคหลอดเลือดสมอง<br>"
                f"โอกาสเสี่ยง ≈ {prob:.1f}%"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box' style='background:#d4edda;color:#155724;'>"
                f"✅ ความเสี่ยงต่ำ<br>"
                f"โอกาสเสี่ยง ≈ {prob:.1f}%"
                f"</div>",
                unsafe_allow_html=True
            )

    except:
        st.warning("⚠️ กรุณากรอกข้อมูลให้ครบและเป็นตัวเลข")

# =========================
# Footer
# =========================
st.write("")
st.caption("⚕️ ระบบนี้เป็นเพียงการประเมินเบื้องต้น ไม่ใช่วินิจฉัยทางการแพทย์")
