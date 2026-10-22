import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. 페이지 설정 및 주의사항 경고 문구 표시
st.set_page_config(page_title="패혈증 위험도 예측 데모", page_icon="🏥")
st.title("🏥 패혈증(Sepsis) 위험도 예측 데모")
st.warning("⚠️ 본 시스템은 연구 및 데모용이며 실제 진단에 사용할 수 없습니다.")

st.divider()

# 2. 모델 및 피처 목록 로드
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

# 3. 주요 수치 입력칸 (요구사항 12개 변수)
st.subheader("📋 주요 수치 입력")

col1, col2 = st.columns(2)

with col1:
    hr = st.number_input("HR (심박수)", value=80.0)
    o2sat = st.number_input("O2Sat (산소포화도)", value=98.0)
    temp = st.number_input("Temp (체온)", value=36.5)
    sbp = st.number_input("SBP (수축기혈압)", value=120.0)
    map_val = st.number_input("MAP (평균혈압)", value=85.0)
    dbp = st.number_input("DBP (이완기혈압)", value=70.0)

with col2:
    resp = st.number_input("Resp (호흡수)", value=18.0)
    lactate = st.number_input("Lactate (젖산)", value=1.0)
    creatinine = st.number_input("Creatinine (크레아티닌)", value=0.9)
    wbc = st.number_input("WBC (백혈구)", value=7.5)
    platelets = st.number_input("Platelets (혈소판)", value=250.0)
    age = st.number_input("Age (나이)", value=50, step=1)

# 4. 위험도 예측 실행
if st.button("위험도 예측", use_container_width=True):
    user_inputs = {
        'HR': hr, 'O2Sat': o2sat, 'Temp': temp, 'SBP': sbp, 'MAP': map_val,
        'DBP': dbp, 'Resp': resp, 'Lactate': lactate, 'Creatinine': creatinine,
        'WBC': wbc, 'Platelets': platelets, 'Age': age
    }

    # 요구사항: 134개 피처를 정해진 FEATURES 순서대로 생성 (미측정 변수 기본 0)
    input_data = pd.DataFrame(0.0, index=[0], columns=FEATURES)

    for col in FEATURES:
        prefix = col.split('_')[0]
        
        # 입력한 12개 주요 변수 관련 피처 처리
        if prefix in user_inputs:
            if col.endswith('_missing'):
                input_data.loc[0, col] = 0.0  # 측정한 피처이므로 missing = 0
            else:
                input_data.loc[0, col] = float(user_inputs[prefix])
        # 미입력 피처 처리 (요구사항: 미측정 피처는 0으로 채우기 / missing 플래그는 1)
        else:
            if col.endswith('_missing'):
                input_data.loc[0, col] = 1.0  # 미측정 상태 = 1
            else:
                input_data.loc[0, col] = 0.0  # 미측정 피처 수치 = 0

    input_data = input_data.astype(float)

    # 모델 예측 (고위험일 확률 계산)
    risk_prob = model.predict_proba(input_data)[0][1]
    risk_percentage = risk_prob * 100

    # 결과 및 위험도별 색상 출력
    st.subheader("예측 결과")
    if risk_percentage >= 70:
        st.error(f"🚨 위험도: {risk_percentage:.1f}% (고위험)")
    elif risk_percentage >= 30:
        st.warning(f"🟧 위험도: {risk_percentage:.1f}% (중등도 위험)")
    else:
        st.success(f"🟢 위험도: {risk_percentage:.1f}% (저위험)")
