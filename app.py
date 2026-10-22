import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="패혈증 위험도 예측 데모", page_icon="🏥")
st.title("🏥 패혈증(Sepsis) 위험도 예측 데모")
st.warning("⚠️ 본 시스템은 연구 및 데모용이며 실제 진단에 사용할 수 없습니다.")

st.divider()

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

with st.expander("🔍 모델 피처(FEATURES) 정보 확인하기"):
    st.write(f"총 피처 개수: {len(FEATURES)}")
    st.write("피처 목록:", FEATURES)

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

if st.button("위험도 예측", use_container_width=True):
    input_dict = {
        'HR': hr, 'O2Sat': o2sat, 'Temp': temp, 'SBP': sbp, 'MAP': map_val,
        'DBP': dbp, 'Resp': resp, 'Lactate': lactate, 'Creatinine': creatinine,
        'WBC': wbc, 'Platelets': platelets, 'Age': age
    }

    # 134개 피처 데이터프레임 생성 (기본값 NaN)
    input_data = pd.DataFrame(index=[0], columns=FEATURES)

    # 134개 피처 전체를 순회하면서, 입력 변수명이 포함되어 있으면 입력값을 할당
    for f in FEATURES:
        matched = False
        for k, v in input_dict.items():
            # 완전 일치하거나 피처 이름에 입력 키워드가 포함된 경우 (예: HR_mean 등)
            if k == f or k in f:
                input_data.loc[0, f] = float(v)
                matched = True
                break
        if not matched:
            input_data.loc[0, f] = np.nan

    # 숫자형 데이터로 변환
    input_data = input_data.astype(float)

    # 예측 실행
    risk_prob = model.predict_proba(input_data)[0][1]
    risk_percentage = risk_prob * 100

    st.subheader("예측 결과")
    if risk_percentage >= 70:
        st.error(f"🚨 위험도: {risk_percentage:.1f}% (고위험)")
    elif risk_percentage >= 30:
        st.warning(f"🟧 위험도: {risk_percentage:.1f}% (중등도 위험)")
    else:
        st.success(f"🟢 위험도: {risk_percentage:.1f}% (저위험)")
