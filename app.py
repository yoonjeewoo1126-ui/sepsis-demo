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

# --- 디버깅용: features.pkl 정보 확인 ---
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
    # 입력 수치 사전 구성
    input_dict = {
        'HR': hr, 'O2Sat': o2sat, 'Temp': temp, 'SBP': sbp, 'MAP': map_val,
        'DBP': dbp, 'Resp': resp, 'Lactate': lactate, 'Creatinine': creatinine,
        'WBC': wbc, 'Platelets': platelets, 'Age': age
    }

    # 모든 피처를 기본 0 또는 NaN 대신 입력값 매핑 후 missing 처리
    input_df = pd.DataFrame([input_dict])
    
    # FEATURES에 정의된 컬럼 구조 생성
    input_data = pd.DataFrame(columns=FEATURES)
    
    # 매핑 가능한 입력 데이터 채우기
    for col in FEATURES:
        if col in input_dict:
            input_data.loc[0, col] = input_dict[col]
        else:
            # 입력값에 없는 피처는 NaN 처리 (LightGBM 결측치 자동 처리 기능 활용)
            input_data.loc[0, col] = np.nan

    # 데이터 타입 숫자형으로 명시적 변환
    input_data = input_data.astype(float)

    # 위험도 예측
    risk_prob = model.predict_proba(input_data)[0][1]
    risk_percentage = risk_prob * 100

    st.subheader("예측 결과")
    if risk_percentage >= 70:
        st.error(f"🚨 위험도: {risk_percentage:.1f}% (고위험)")
    elif risk_percentage >= 30:
        st.warning(f"🟧 위험도: {risk_percentage:.1f}% (중등도 위험)")
    else:
        st.success(f"🟢 위험도: {risk_percentage:.1f}% (저위험)")
