import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. 페이지 기본 설정 & 필수 연구용 경고 문구
st.set_page_config(page_title="패혈증 위험도 예측 데모", page_icon="🏥", layout="centered")
st.title("🏥 패혈증(Sepsis) 위험도 예측 데모")
st.warning("⚠️ **주의사항**: 본 시스템은 연구 및 데모용이며, 실제 의료 진단 및 치료 목적으로 사용할 수 없습니다.")

st.divider()

# 2. 모델 및 피처 목록 로드
@st.cache_resource
def load_assets():
    model = joblib.load('lgbm_model.pkl')
    features = joblib.load('features.pkl')
    return model, features

try:
    model, FEATURES = load_assets()
except Exception as e:
    st.error(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 주요 12개 수치 입력칸 (드래그식 슬라이더)
st.subheader("📋 주요 임상 수치 입력")
st.caption("슬라이더를 마우스로 끌거나 숫자를 직접 입력해 보세요.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🫀 활력징후 (Vitals)")
    hr = st.slider("심박수 (HR)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
    o2sat = st.slider("산소포화도 (O2Sat, %)", min_value=0.0, max_value=100.0, value=98.0, step=0.5)
    temp = st.slider("체온 (Temp, °C)", min_value=30.0, max_value=45.0, value=36.5, step=0.1)
    sbp = st.slider("수축기 혈압 (SBP)", min_value=0.0, max_value=240.0, value=120.0, step=1.0)
    map_val = st.slider("평균 혈압 (MAP)", min_value=0.0, max_value=200.0, value=85.0, step=1.0)
    dbp = st.slider("이완기 혈압 (DBP)", min_value=0.0, max_value=150.0, value=70.0, step=1.0)

with col2:
    st.markdown("### 🧪 혈액검사 & 기타")
    resp = st.slider("호흡수 (Resp)", min_value=0.0, max_value=60.0, value=18.0, step=1.0)
    lactate = st.slider("젖산 (Lactate)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
    creatinine = st.slider("크레아티닌 (Creatinine)", min_value=0.0, max_value=15.0, value=0.9, step=0.1)
    wbc = st.slider("백혈구 수 (WBC)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
    platelets = st.slider("혈소판 수 (Platelets)", min_value=0.0, max_value=800.0, value=250.0, step=5.0)
    age = st.slider("나이 (Age)", min_value=0, max_value=120, value=50, step=1)

st.divider()

# 4. 버튼 누르면 예측 수행
if st.button("🚀 위험도 예측하기", use_container_width=True):
    input_dict = {
        'HR': hr, 'O2Sat': o2sat, 'Temp': temp, 'SBP': sbp, 'MAP': map_val,
        'DBP': dbp, 'Resp': resp, 'Lactate': lactate, 'Creatinine': creatinine,
        'WBC': wbc, 'Platelets': platelets, 'Age': age
    }

    # 미측정 피처는 기본값 0으로 채우기
    input_data = pd.DataFrame(0, index=[0], columns=FEATURES)

    # 입력받은 수치 채우기
    for col, val in input_dict.items():
        if col in input_data.columns:
            input_data[col] = val

    # FEATURES 목록 순서 그대로 정렬
    input_data = input_data[FEATURES]

    # 위험도 예측
    try:
        risk_prob = model.predict_proba(input_data)[0][1]
        risk_percentage = risk_prob * 100

        st.subheader("📊 예측 결과")
        if risk_percentage >= 70:
            st.error(f"🚨 **고위험군**: 패혈증 위험도 **{risk_percentage:.1f}%**")
            st.progress(risk_prob)
        elif risk_percentage >= 30:
            st.warning(f"🟧 **중등도 위험군**: 패혈증 위험도 **{risk_percentage:.1f}%**")
            st.progress(risk_prob)
        else:
            st.success(f"🟢 **저위험군**: 패혈증 위험도 **{risk_percentage:.1f}%**")
            st.progress(risk_prob)

    except Exception as e:
        st.error(f"예측 수행 중 오류가 발생했습니다: {e}")
