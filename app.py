import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. 페이지 설정 및 경고 문구
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

# 3. 주요 수치 입력 12개
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

# 4. 버튼 누르면 위험도 계산하기
if st.button("위험도 예측", use_container_width=True):
    # 입력된 12개 매핑 사전
    user_inputs = {
        'HR': hr,
        'O2Sat': o2sat,
        'Temp': temp,
        'SBP': sbp,
        'MAP': map_val,
        'DBP': dbp,
        'Resp': resp,
        'Lactate': lactate,
        'Creatinine': creatinine,
        'WBC': wbc,
        'Platelets': platelets,
        'Age': age
    }

    # 134개 피처를 담을 입력 데이터프레임 초기화 (기본 NaN)
    input_data = pd.DataFrame(np.nan, index=[0], columns=FEATURES)

    # 134개 피처 자동 매핑 로직
    for col in FEATURES:
        # 1) 12개 핵심 변수 및 관련 파생 피처 처리 (HR, HR_mean6, HR_max6, HR_min6, HR_d1 등)
        matched_key = None
        for key in user_inputs.keys():
            if col == key or col.startswith(key + '_'):
                matched_key = key
                break
        
        if matched_key:
            if col.endswith('_missing'):
                input_data.loc[0, col] = 0  # 입력된 값이므로 결측 플래그 = 0
            else:
                input_data.loc[0, col] = float(user_inputs[matched_key])
        else:
            # 2) 사용자가 입력하지 않은 기타 피처 처리
            if col.endswith('_missing'):
                input_data.loc[0, col] = 1  # 입력 안 된 측정치 = 1 (결측 처리)

    # 숫자형 타입 보장
    input_data = input_data.astype(float)

    # 위험도 예측 (확률 산출)
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
