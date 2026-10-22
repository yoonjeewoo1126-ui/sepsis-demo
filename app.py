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

if st.button("위험도 예측", use_container_width=True):
    user_inputs = {
        'HR': hr, 'O2Sat': o2sat, 'Temp': temp, 'SBP': sbp, 'MAP': map_val,
        'DBP': dbp, 'Resp': resp, 'Lactate': lactate, 'Creatinine': creatinine,
        'WBC': wbc, 'Platelets': platelets, 'Age': age
    }

    input_data = pd.DataFrame(0.0, index=[0], columns=FEATURES)

    for col in FEATURES:
        # 단일 수치(HR, Temp 등) 및 평균/최대/최소 수치에만 값 할당
        prefix = col.split('_')[0]
        suffix = col.replace(prefix + '_', '') if '_' in col else ''

        if prefix in user_inputs:
            if col.endswith('_missing'):
                input_data.loc[0, col] = 0.0
            elif suffix in ['', 'mean6', 'max6', 'min6']:
                input_data.loc[0, col] = float(user_inputs[prefix])
            else:
                # _d1, _std6 등 파생 변수는 0 처리
                input_data.loc[0, col] = 0.0
        else:
            if col.endswith('_missing'):
                input_data.loc[0, col] = 1.0
            else:
                input_data.loc[0, col] = 0.0

    input_data = input_data.astype(float)

    # 확률 계산
    probs = model.predict_proba(input_data)[0]
    risk_prob = probs[1]
    risk_percentage = risk_prob * 100

    st.subheader("예측 결과")
    st.write(f"📊 **원시 위험도 확률 (Class 1):** `{risk_prob:.4f}` ({risk_percentage:.2f}%)")

    if risk_percentage >= 70:
        st.error(f"🚨 위험도: {risk_percentage:.1f}% (고위험)")
    elif risk_percentage >= 30:
        st.warning(f"🟧 위험도: {risk_percentage:.1f}% (중등도 위험)")
    else:
        st.success(f"🟢 위험도: {risk_percentage:.1f}% (저위험)")
