import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. 연구용 경고 문구 표시
st.set_page_config(page_title="패혈증 위험도 예측 데모", page_icon="🏥")
st.title("🏥 패혈증(Sepsis) 위험도 예측 데모")
st.warning("⚠️ 본 시스템은 연구 및 데모용이며 실제 진단에 사용할 수 없습니다.")

st.divider()

# 2. 모델 및 피처 목록 가져오기
@st.cache_resource
def load_assets():
    model = joblib.load('lgbm_model.pkl')
    FEATURES = joblib.load('features.pkl')
    return model, FEATURES

try:
    model, FEATURES = load_assets()
except Exception as e:
    st.error(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 입력하지 않은 일반 피처를 채울 값 (0 또는 -1)
MISSING_FILL = 0

# 3. 주요 수치 입력칸 12개 만들기
st.subheader("📋 주요 수치 입력")

col1, col2 = st.columns(2)

with col1:
    hr = st.number_input("HR", value=80.0)
    o2sat = st.number_input("O2Sat", value=98.0)
    temp = st.number_input("Temp", value=36.5)
    sbp = st.number_input("SBP", value=120.0)
    map_val = st.number_input("MAP", value=85.0)
    dbp = st.number_input("DBP", value=70.0)

with col2:
    resp = st.number_input("Resp", value=18.0)
    lactate = st.number_input("Lactate", value=1.0)
    creatinine = st.number_input("Creatinine", value=0.9)
    wbc = st.number_input("WBC", value=7.5)
    platelets = st.number_input("Platelets", value=250.0)
    age = st.number_input("Age", value=50, step=1)

# 4. 버튼 누르면 위험도 계산하기
if st.button("위험도 예측", use_container_width=True):
    # 입력한 수치 정리
    input_dict = {
        'HR': hr, 'O2Sat': o2sat, 'Temp': temp, 'SBP': sbp, 'MAP': map_val,
        'DBP': dbp, 'Resp': resp, 'Lactate': lactate, 'Creatinine': creatinine,
        'WBC': wbc, 'Platelets': platelets, 'Age': age
    }

    # 일단 전부 기본값으로 채우기
    input_data = pd.DataFrame(float(MISSING_FILL), index=[0], columns=FEATURES)

    for name, val in input_dict.items():
        # 기본 수치 + 평균/최대/최소(6시간)는 입력값 그대로
        for suffix in ['', '_mean6', '_max6', '_min6']:
            c = name + suffix
            if c in input_data.columns:
                input_data[c] = float(val)
        # 변동량(표준편차, 1시간 변화)은 0
        for suffix in ['_std6', '_d1']:
            c = name + suffix
            if c in input_data.columns:
                input_data[c] = 0.0

    # 결측 표시 피처: 입력한 항목은 0(측정함), 안 한 항목은 1(측정 안 함)
    for c in FEATURES:
        if c.endswith('_missing'):
            base = c[:-len('_missing')]
            input_data[c] = 0.0 if base in input_dict else 1.0

    # FEATURES 목록 순서 그대로 맞추기
    input_data = input_data[FEATURES]

    # 위험도 예측
    risk_prob = model.predict_proba(input_data)[0][1]
    risk_percentage = risk_prob * 100

    # 결과 및 색상 출력
    st.subheader("예측 결과")
    if risk_percentage >= 70:
        st.error(f"🚨 위험도: {risk_percentage:.1f}% (고위험)")
    elif risk_percentage >= 30:
        st.warning(f"🟧 위험도: {risk_percentage:.1f}% (중등도 위험)")
    else:
        st.success(f"🟢 위험도: {risk_percentage:.1f}% (저위험)")
