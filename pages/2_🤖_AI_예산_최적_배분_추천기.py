import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI 예산 최적 배분 추천기", layout="wide")


@st.cache_resource
def load_assets():
  df = pd.read_csv("data/processed/merged_care_dataset.csv")
  model = joblib.load("model/policy_effect_model.joblib")
  scaler = joblib.load("model/feature_scaler.joblib")
  return df, model, scaler


df, model, scaler = load_assets()

st.title("🤖 AI 기반 지자체 예산 최적 배분 추천기")
st.markdown(
    "한정된 추가 예산 한도 내에서 **위험 지수를 최대로 낮출 수 있는 인력 증원 및 시설 확충의 황금 비율**을 AI가 탐색합니다."
)
st.divider()

# 입력 설정 영역
col1, col2 = st.columns([1, 2])

with col1:
  st.subheader("⚙️ 조건 설정")
  target_region = st.selectbox("대상 자치구 선택", df["region_name"].unique())
  total_budget = st.number_input(
      "가용 추가 예산 한도 (억 원)",
      min_value=1.0,
      max_value=50.0,
      value=10.0,
      step=1.0,
  )
  worker_cost = st.number_input(
      "생활지원사 1명당 연간 소요 예산 (억 원)",
      min_value=0.1,
      max_value=2.0,
      value=0.4,
      step=0.1,
  )
  facility_cost = st.number_input(
      "돌봄 거점 1개소 확충 예산 (억 원)",
      min_value=0.5,
      max_value=10.0,
      value=2.5,
      step=0.5,
  )
  run_btn = st.button(
      "🚀 AI 최적 조합 탐색 실행", type="primary", use_container_width=True
  )

with col2:
  st.subheader("🎯 탐색 결과 및 최적 정책 제안")
  region_row = df[df["region_name"] == target_region].iloc[0]

  base_features = np.array([[
      region_row["elderly_ratio"],
      region_row["single_household_ratio"],
      region_row["welfare_facilities"],
      region_row["care_workers"],
      region_row["annual_budget"],
  ]])
  base_risk = model.predict(scaler.transform(base_features))[0]

  if run_btn:
    candidates = []
    max_facilities = int(total_budget // facility_cost)

    # 예산 한도 내 가능한 모든 (시설 수, 인력 수) 조합 탐색 (Grid Search)
    for add_fac in range(max_facilities + 1):
      rem_budget_for_workers = total_budget - (add_fac * facility_cost)
      add_workers = int(rem_budget_for_workers // worker_cost)
      used_budget = round(
          add_fac * facility_cost + add_workers * worker_cost, 2
      )

      sim_features = np.array([[
          region_row["elderly_ratio"],
          region_row["single_household_ratio"],
          region_row["welfare_facilities"] + add_fac,
          region_row["care_workers"] + add_workers,
          region_row["annual_budget"] + used_budget,
      ]])
      pred_risk = model.predict(scaler.transform(sim_features))[0]
      reduction = base_risk - pred_risk

      candidates.append({
          "돌봄 거점 확충 (개)": add_fac,
          "생활지원사 증원 (명)": add_workers,
          "실제 집행 예산 (억 원)": used_budget,
          "예측 위험 지수 (점)": round(pred_risk, 2),
          "위험도 감소폭 (점)": round(reduction, 2),
      })

    res_df = pd.DataFrame(candidates).sort_values(
        by="위험도 감소폭 (점)", ascending=False
    )
    best = res_df.iloc[0]

    st.success(
        f"**[{target_region}] 최적 정책 조합 도출 완료!** (현재 위험 지수:"
        f" {base_risk:.1f}점 $\\rightarrow$ 개선 후:"
        f" **{best['예측 위험 지수 (점)']}점**)"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("추천 거점 시설 확충", f"+{int(best['돌봄 거점 확충 (개)'])} 개소")
    m2.metric(
        "추천 생활지원사 증원", f"+{int(best['생활지원사 증원 (명)'])} 명"
    )
    m3.metric("최대 위험 감소폭", f"-{best['위험도 감소폭 (점)']} 점")

    st.markdown("#### 📊 예산 한도 내 조합별 시뮬레이션 순위")
    st.dataframe(res_df.reset_index(drop=True), use_container_width=True)
  else:
    st.info(
        "좌측에서 예산 조건을 설정한 뒤 **[AI 최적 조합 탐색 실행]** 버튼을 눌러주세요."
    )