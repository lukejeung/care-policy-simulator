import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="자치구별 상세 비교 및 우선순위", layout="wide")


@st.cache_data
def load_data():
  return pd.read_csv("data/processed/merged_care_dataset.csv")


df = load_data()

st.title("📊 서울시 25개 자치구 돌봄 위험도 비교 및 우선순위 분석")
st.markdown(
    "고령인구 비율과 현재 돌봄 인프라를 종합하여 **선제적 복지 예산 투입이 시급한 우선순위 자치구**를 도출합니다."
)
st.divider()

# 1. 긴급 지원 우선순위 TOP 5 카드 표시
st.subheader("🚨 돌봄 공백·고독사 위험 지수 TOP 5 자치구")
top5 = df.sort_values(by="risk_score", ascending=False).head(5)

cols = st.columns(5)
for idx, (_, row) in enumerate(top5.iterrows()):
  cols[idx].metric(
      label=f"{idx + 1}위 · {row['region_name']}",
      value=f"{row['risk_score']:.1f} 점",
      delta=f"고령비율 {row['elderly_ratio']:.1f}%",
      delta_color="inverse",
  )

st.divider()

# 2. 두 자치구 1:1 비교 분석
st.subheader("⚖️ 자치구 간 1:1 돌봄 환경 비교")
col_sel1, col_sel2 = st.columns(2)

region_a = col_sel1.selectbox(
    "기준 자치구 (A)", df["region_name"].unique(), index=0
)
region_b = col_sel2.selectbox(
    "비교 자치구 (B)", df["region_name"].unique(), index=8
)  # 예: 강북구

data_a = df[df["region_name"] == region_a].iloc[0]
data_b = df[df["region_name"] == region_b].iloc[0]

compare_df = pd.DataFrame({
    "분석 지표": [
        "65세 이상 고령인구 비율 (%)",
        "1인 가구 비율 (%)",
        "복지관/돌봄센터 수 (개)",
        "생활지원사 인력 (명)",
        "연간 돌봄 예산 (억 원)",
        "종합 위험 지수 (점)",
    ],
    region_a: [
        round(data_a["elderly_ratio"], 2),
        round(data_a["single_household_ratio"], 2),
        int(data_a["welfare_facilities"]),
        int(data_a["care_workers"]),
        round(data_a["annual_budget"], 2),
        round(data_a["risk_score"], 2),
    ],
    region_b: [
        round(data_b["elderly_ratio"], 2),
        round(data_b["single_household_ratio"], 2),
        int(data_b["welfare_facilities"]),
        int(data_b["care_workers"]),
        round(data_b["annual_budget"], 2),
        round(data_b["risk_score"], 2),
    ],
})

st.table(compare_df)

# 3. 전체 자치구 위험도 순위표 및 CSV 다운로드 기능
st.subheader("📋 서울시 전체 자치구 위험도 랭킹 데이터")
ranked_df = df.sort_values(by="risk_score", ascending=False).reset_index(
    drop=True
)
ranked_df.index = ranked_df.index + 1
ranked_df.index.name = "위험순위"

st.dataframe(
    ranked_df[[
        "region_name",
        "elderly_ratio",
        "single_household_ratio",
        "welfare_facilities",
        "care_workers",
        "annual_budget",
        "risk_score",
    ]],
    use_container_width=True,
)

# pages/1_📊_자치구별_상세_비교_및_우선순위.py 내 ranked_df 생성 직후에 추가
def get_risk_grade(score):
  if score >= 32:
    return "🔴 고위험"
  elif score >= 20:
    return "🟠 주의"
  else:
    return "🟢 양호"


ranked_df["위험등급"] = ranked_df["risk_score"].apply(get_risk_grade)

csv_data = ranked_df.to_csv(index=True).encode("utf-8-sig")
st.download_button(
    label="📥 전체 자치구 분석 결과 CSV 다운로드",
    data=csv_data,
    file_name="seoul_care_risk_ranking.csv",
    mime="text/csv",
)