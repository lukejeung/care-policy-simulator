import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="AI 맞춤 돌봄 정책 시뮬레이터", layout="wide")


# 데이터 및 모델 로드 (캐싱으로 속도 최적화)
@st.cache_resource
def load_assets():
  df = pd.read_csv("data/processed/merged_care_dataset.csv")
  model = joblib.load("model/policy_effect_model.joblib")
  scaler = joblib.load("model/feature_scaler.joblib")
  return df, model, scaler


df, model, scaler = load_assets()

# --- 사이드바: 지역 선택 및 시나리오 설정 ---
st.sidebar.header("🏛️ 정책 시나리오 설정")
selected_region = st.sidebar.selectbox(
    "분석 대상 행정동 선택", df["region_name"].unique()
)

# 선택된 지역의 현재 데이터 추출
region_data = df[df["region_name"] == selected_region].iloc[0]

st.sidebar.subheader("투입 자원 시뮬레이션 (What-If)")
add_budget = st.sidebar.slider("추가 돌봄 예산 투입 (억 원)", 0.0, 10.0, 2.0, 0.5)
add_workers = st.sidebar.slider("생활지원사 인력 증원 (명)", 0, 30, 5, 1)
add_facilities = st.sidebar.slider("돌봄 거점/복지관 확충 (개)", 0, 5, 0, 1)

# --- 메인 대시보드 화면 ---
st.title("📊 AI 기반 지역별 맞춤 돌봄 정책 효과 시뮬레이션")
st.markdown(
    f"**선택 지역:** `{selected_region}` (행정동 코드:"
    f" `{region_data['admin_code']}`)"
)
st.divider()

# 1. 현재 지역 기초 현황
col1, col2, col3, col4 = st.columns(4)
col1.metric("고령인구 비율", f"{region_data['elderly_ratio']:.1f}%")
col2.metric("1인 가구 비율", f"{region_data['single_household_ratio']:.1f}%")
col3.metric("현재 돌봄 인력", f"{int(region_data['care_workers'])}명")
col4.metric("현재 연간 예산", f"{region_data['annual_budget']:.1f}억 원")

# 2. 시뮬레이션 예측 계산
# 기존 상태 입력값
current_features = np.array([[
    region_data["elderly_ratio"],
    region_data["single_household_ratio"],
    region_data["welfare_facilities"],
    region_data["care_workers"],
    region_data["annual_budget"],
]])
current_risk = model.predict(scaler.transform(current_features))[0]

# 정책 투입 후 입력값
simulated_features = np.array([[
    region_data["elderly_ratio"],
    region_data["single_household_ratio"],
    region_data["welfare_facilities"] + add_facilities,
    region_data["care_workers"] + add_workers,
    region_data["annual_budget"] + add_budget,
]])
simulated_risk = model.predict(scaler.transform(simulated_features))[0]

risk_diff = simulated_risk - current_risk
reduction_rate = (abs(risk_diff) / current_risk) * 100 if risk_diff < 0 else 0

# 3. 시뮬레이션 결과 비교 출력
st.subheader("🔮 정책 투입 후 예측 결과")
res_col1, res_col2, res_col3 = st.columns(3)

res_col1.metric("현재 예측 위험 지수", f"{current_risk:.1f} 점")
res_col2.metric(
    "정책 적용 후 위험 지수",
    f"{simulated_risk:.1f} 점",
    delta=f"{risk_diff:.1f} 점",
    delta_color="inverse",  # 점수가 내려갈수록 초록색(긍정적) 표시
)
res_col3.metric("고독사·돌봄 공백 위험 감소율", f"{reduction_rate:.1f}%")

# 4. 정책 분석 코멘트 및 데이터 표
if risk_diff < -3.0:
  st.success(
      f"✅ **효과 우수:** 해당 시나리오 적용 시 위험 지수가 **{abs(risk_diff):.1f}점** 큰 폭으로 개선될 것으로 예측됩니다."
  )
elif risk_diff < 0:
  st.info(
      f"ℹ️ **점진적 개선:** 위험 지수가 **{abs(risk_diff):.1f}점** 완화됩니다. 예산 대비 효율을 높이기 위해 인력 증원 비율을 조정해 보세요."
  )
else:
  st.warning("⚠️ 추가 자원 투입량을 늘려 시뮬레이션을 다시 실행해 보세요.")

with st.expander("전체 행정동 데이터 미리보기"):
  st.dataframe(df)

import folium
from streamlit_folium import st_folium

st.divider()
st.subheader("🗺️ 서울시 25개 자치구 돌봄 위험도 GIS 현황판")

# 서울시 25개 자치구 실제 중심 좌표(위도, 경도) 딕셔너리
seoul_coords = {
    "종로구": (37.5730, 126.9794),
    "중구": (37.5641, 126.9979),
    "용산구": (37.5326, 126.9905),
    "성동구": (37.5633, 127.0371),
    "광진구": (37.5385, 127.0823),
    "동대문구": (37.5744, 127.0400),
    "중랑구": (37.6065, 127.0927),
    "성북구": (37.5891, 127.0182),
    "강북구": (37.6396, 127.0257),
    "도봉구": (37.6688, 127.0471),
    "노원구": (37.6542, 127.0568),
    "은평구": (37.6027, 126.9291),
    "서대문구": (37.5791, 126.9368),
    "마포구": (37.5663, 126.9014),
    "양천구": (37.5169, 126.8664),
    "강서구": (37.5510, 126.8495),
    "구로구": (37.4954, 126.8874),
    "금천구": (37.4568, 126.8954),
    "영등포구": (37.5264, 126.8962),
    "동작구": (37.5124, 126.9393),
    "관악구": (37.4784, 126.9516),
    "서초구": (37.4837, 127.0324),
    "강남구": (37.5172, 127.0473),
    "송파구": (37.5145, 127.1059),
    "강동구": (37.5301, 127.1238),
}

m = folium.Map(
    location=[37.5665, 126.9780], zoom_start=11, tiles="OpenStreetMap"
)

for _, row in df.iterrows():
  region = row["region_name"]
  if region not in seoul_coords:
    continue

  lat, lon = seoul_coords[region]
  score = simulated_risk if region == selected_region else row["risk_score"]

  if score >= 60:
    color = "#e74c3c"  # 고위험 (빨강)
  elif score >= 45:
    color = "#f39c12"  # 주의 (주황)
  else:
    color = "#2ecc71"  # 양호 (초록)

  folium.CircleMarker(
      location=[lat, lon],
      radius=score / 3.5,
      color=color,
      fill=True,
      fill_color=color,
      fill_opacity=0.65,
      popup=(
          f"<b>{region}</b><br>고령비율:"
          f" {row['elderly_ratio']:.1f}%<br>위험지수: {score:.1f}점"
      ),
  ).add_to(m)

st_folium(m, width=1000, height=480)

st.divider()
st.subheader("🤖 AI 기반 예산 배분 시나리오 비교 (ROI 분석)")

# 3가지 대표적인 정책 대안 정의 (동일 예산 3억 원 투입 가정)
scenarios = [
    {
        "name": "대안 A: 생활지원사 인력 집중 투입 (인력 +10명)",
        "budget": 3.0,
        "workers": 10,
        "fac": 0,
    },
    {
        "name": "대안 B: 소규모 돌봄 거점 확충 (시설 +2개소)",
        "budget": 3.0,
        "workers": 2,
        "fac": 2,
    },
    {
        "name": "대안 C: 인력·인프라 균형 배분 (인력 +5명, 시설 +1개소)",
        "budget": 3.0,
        "workers": 5,
        "fac": 1,
    },
]

results = []
for sc in scenarios:
  sc_features = np.array([[
      region_data["elderly_ratio"],
      region_data["single_household_ratio"],
      region_data["welfare_facilities"] + sc["fac"],
      region_data["care_workers"] + sc["workers"],
      region_data["annual_budget"] + sc["budget"],
  ]])
  pred_risk = model.predict(scaler.transform(sc_features))[0]
  drop = current_risk - pred_risk
  results.append({
      "정책 시나리오": sc["name"],
      "투입 예산": f"{sc['budget']}억 원",
      "예상 위험 지수": f"{pred_risk:.1f} 점",
      "위험도 감소폭": f"-{drop:.2f} 점",
      "예산 1억당 감소 효율": f"{drop / sc['budget']:.2f} 점/억",
  })

res_df = pd.DataFrame(results)
st.table(res_df)