from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[6]
LOAD_CSV_DIR = PROJECT_ROOT / "01_notebooks" / "03_models"

if not LOAD_CSV_DIR.exists():
    raise FileNotFoundError(f"load_csv.py 폴더를 찾을 수 없습니다: {LOAD_CSV_DIR}")

if str(LOAD_CSV_DIR) not in sys.path:
    sys.path.append(str(LOAD_CSV_DIR))

from load_csv import create_inference_data


FEATURE_COLUMNS = [
    "age",
    "plan_tier",
    "monthly_spend",
    "age_group",
    "subscription_tenure_days",
    "watch_count",
    "unique_movies",
    "total_watch_time",
    "avg_watch_time",
    "watch_days",
    "recent_watch_count",
    "days_since_last_watch",
    "avg_progress",
    "completion_rate",
    "download_ratio",
    "avg_rating",
]

DISPLAY_NAME_MAP = {
    "age": "연령",
    "plan_tier": "요금제 등급",
    "monthly_spend": "월 지출액",
    "age_group": "연령대",
    "subscription_tenure_days": "구독 유지 기간",
    "watch_count": "총 시청 횟수",
    "unique_movies": "콘텐츠 다양성",
    "total_watch_time": "총 시청 시간",
    "avg_watch_time": "평균 시청 시간",
    "watch_days": "시청 일수",
    "recent_watch_count": "최근 시청 횟수",
    "days_since_last_watch": "최근 미시청 일수",
    "avg_progress": "평균 시청 진행률",
    "completion_rate": "완주율",
    "download_ratio": "다운로드 비율",
    "avg_rating": "평균 평점",
}


def load_uploaded_raw_dataframe(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("업로드된 파일이 없습니다.")

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx"):
        raw_bytes = uploaded_file.getvalue()
        return pd.read_excel(BytesIO(raw_bytes))

    raise ValueError("csv 또는 xlsx 파일만 지원합니다.")


def prepare_inference_dataframe(uploaded_file) -> pd.DataFrame:
    raw_user_df = load_uploaded_raw_dataframe(uploaded_file)
    inference_df = create_inference_data(raw_user_df)

    if "is_active" in inference_df.columns:
        inference_df = inference_df.drop(columns=["is_active"])

    missing_columns = [col for col in FEATURE_COLUMNS if col not in inference_df.columns]
    if missing_columns:
        raise ValueError(
            "전처리 후에도 모델 입력 컬럼이 부족합니다. "
            f"누락 컬럼: {', '.join(missing_columns)}"
        )

    for col in FEATURE_COLUMNS:
        inference_df[col] = pd.to_numeric(inference_df[col], errors="coerce")

    null_mask = inference_df[FEATURE_COLUMNS].isnull().any()
    if null_mask.any():
        null_cols = null_mask[null_mask].index.tolist()
        raise ValueError(
            "모델 입력 컬럼에 결측치가 있습니다. "
            f"문제 컬럼: {', '.join(null_cols)}"
        )

    return inference_df


def load_xgb_model(model_path: str) -> XGBClassifier:
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file}")

    model = XGBClassifier()
    model.load_model(str(model_file))
    return model


def label_risk(prob: float) -> str:
    if prob >= 0.75:
        return "높은 위험"
    if prob >= 0.50:
        return "중간 위험"
    if prob >= 0.25:
        return "낮은 위험"
    return "안전"


def predict_with_model(model: XGBClassifier, inference_df: pd.DataFrame) -> pd.DataFrame:
    result_df = inference_df.copy()
    x_infer = result_df[FEATURE_COLUMNS].copy()

    churn_proba = model.predict_proba(x_infer)[:, 1]
    result_df["churn_probability"] = churn_proba
    result_df["risk_score"] = (churn_proba * 100).round(1)
    result_df["risk_label"] = result_df["churn_probability"].apply(label_risk)

    return result_df


def extract_feature_importance(model: XGBClassifier) -> pd.DataFrame:
    raw_values = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": raw_values,
        }
    ).sort_values("importance", ascending=False)

    total = float(importance_df["importance"].sum())
    if total == 0:
        importance_df["importance_pct"] = 0.0
    else:
        importance_df["importance_pct"] = (
            importance_df["importance"] / total * 100
        ).round(1)

    importance_df["feature_display"] = importance_df["feature"].map(DISPLAY_NAME_MAP)
    return importance_df.reset_index(drop=True)


def make_data_meta(result_df: pd.DataFrame, uploaded_file_name: str, model_name: str = "XGBoost") -> dict:
    total_users = len(result_df)
    high_risk_ratio = round((result_df["churn_probability"] >= 0.75).mean() * 100, 1)
    avg_churn_prob = round(result_df["churn_probability"].mean() * 100, 1)

    return {
        "file_name": uploaded_file_name,
        "record_count": total_users,
        "model_name": model_name,
        "high_risk_ratio": high_risk_ratio,
        "avg_churn_prob": avg_churn_prob,
    }


def make_kpi_data(result_df: pd.DataFrame) -> list[dict]:
    total_users = len(result_df)
    high_risk_users = int((result_df["churn_probability"] >= 0.75).sum())
    high_risk_ratio = float(high_risk_users / max(total_users, 1) * 100)
    predicted_churn_rate = float(result_df["churn_probability"].mean() * 100)
    expected_retention_rate = 100 - predicted_churn_rate

    return [
        {
            "title": "총 사용자",
            "value": f"{total_users:,}",
            "subtext": "업로드 데이터 기준 분석 대상",
            "icon": "👥",
        },
        {
            "title": "고위험 고객",
            "value": f"{high_risk_users:,}명",
            "subtext": f"전체의 {high_risk_ratio:.1f}%",
            "icon": "🚨",
        },
        {
            "title": "평균 예측 이탈확률",
            "value": f"{predicted_churn_rate:.1f}%",
            "subtext": f"예상 유지율 {expected_retention_rate:.1f}%",
            "icon": "📉",
        },
        {
            "title": "예상 유지율",
            "value": f"{expected_retention_rate:.1f}%",
            "subtext": "전체 평균 예측 기준",
            "icon": "📊",
        },
    ]


def make_risk_segments(result_df: pd.DataFrame) -> dict:
    labels = ["높은 위험", "중간 위험", "낮은 위험", "안전"]
    counts = result_df["risk_label"].value_counts()

    return {
        "labels": labels,
        "values": [int(counts.get(label, 0)) for label in labels],
    }


def make_trend_data(result_df: pd.DataFrame) -> dict:
    bins = [-1, 3, 7, 14, 30, 60, 999999]
    labels = ["3일 이하", "4~7일", "8~14일", "15~30일", "31~60일", "61일 이상"]

    trend_df = result_df.copy()
    trend_df["inactive_bucket"] = pd.cut(
        trend_df["days_since_last_watch"],
        bins=bins,
        labels=labels,
    )

    grouped = (
        trend_df.groupby("inactive_bucket", observed=False)
        .agg(
            예측이탈률=("churn_probability", lambda x: round(float(x.mean() * 100), 1) if len(x) > 0 else 0.0),
            평균최근시청횟수=("recent_watch_count", lambda x: round(float(x.mean()), 1) if len(x) > 0 else 0.0),
        )
        .reset_index()
    )

    return {
        "months": grouped["inactive_bucket"].astype(str).tolist(),
        "이탈률": grouped["예측이탈률"].tolist(),
        "활동도": grouped["평균최근시청횟수"].tolist(),
    }


def make_usage_data(result_df: pd.DataFrame) -> dict:
    total_users = max(len(result_df), 1)

    heavy = int((result_df["watch_days"] >= 25).sum())
    regular = int(((result_df["watch_days"] >= 12) & (result_df["watch_days"] < 25)).sum())
    light = int(((result_df["watch_days"] >= 4) & (result_df["watch_days"] < 12)).sum())
    inactive = int((result_df["watch_days"] < 4).sum())

    items = [
        {
            "label": "매우 자주 이용",
            "value": round(heavy / total_users * 100, 1),
            "desc": "충성 유지군",
        },
        {
            "label": "주기적 이용",
            "value": round(regular / total_users * 100, 1),
            "desc": "유지 가능군",
        },
        {
            "label": "가벼운 이용",
            "value": round(light / total_users * 100, 1),
            "desc": "이탈 전환 가능군",
        },
        {
            "label": "거의 미이용",
            "value": round(inactive / total_users * 100, 1),
            "desc": "재활성화 우선군",
        },
    ]

    dominant = max(items, key=lambda x: x["value"])
    insight = f"현재 가장 큰 활동군은 '{dominant['label']}'이며, {dominant['desc']} 성격이 강합니다."

    return {
        "items": items,
        "insight": insight,
    }


def classify_profile(row: pd.Series) -> str:
    if row["days_since_last_watch"] >= 30:
        return "휴면위험형"
    if row["completion_rate"] >= 0.8 and row["watch_days"] >= 15:
        return "몰아보기형"
    if row["unique_movies"] >= 15:
        return "탐색형"
    if row["subscription_tenure_days"] >= 365:
        return "충성형"
    return "라이트시청형"


def make_profile_data(result_df: pd.DataFrame) -> dict:
    profile_df = result_df.copy()
    profile_df["profile_type"] = profile_df.apply(classify_profile, axis=1)
    profile_counts = profile_df["profile_type"].value_counts()

    total = max(len(profile_df), 1)
    items = [
        {"label": label, "value": round(count / total * 100, 1)}
        for label, count in profile_counts.items()
    ]
    items = sorted(items, key=lambda x: x["value"], reverse=True)

    if items:
        insight = f"현재 가장 큰 고객군은 '{items[0]['label']}'이며 전체의 {items[0]['value']:.1f}%를 차지합니다."
    else:
        insight = "시청 프로필 분포를 계산할 수 없습니다."

    return {
        "items": items,
        "insight": insight,
    }


def make_driver_data(importance_df: pd.DataFrame, top_n: int = 5) -> dict:
    top_df = importance_df.head(top_n).copy()

    interpretation_map = {
        "총 시청 시간": "사용량 감소는 가장 강한 이탈 신호입니다.",
        "시청 일수": "정기적 이용 습관이 약해질수록 이탈 위험이 높아집니다.",
        "연령": "연령별로 이탈 패턴 차이가 존재합니다.",
        "연령대": "연령대별 유지 전략을 다르게 가져갈 필요가 있습니다.",
        "요금제 등급": "요금제 민감도에 따라 유지 전략 차별화가 필요합니다.",
        "최근 미시청 일수": "최근 비활성 기간이 길수록 위험도가 상승합니다.",
        "최근 시청 횟수": "최근 활동성 저하는 이탈 가능성과 직접 연결됩니다.",
        "완주율": "콘텐츠 몰입도 저하는 이탈 신호일 수 있습니다.",
    }

    items = [
        {
            "label": row["feature_display"],
            "value": round(float(row["importance_pct"]), 1),
        }
        for _, row in top_df.iterrows()
    ]

    top_name = items[0]["label"] if items else "행동 변화"
    insight = interpretation_map.get(top_name, f"현재 데이터에서는 '{top_name}'의 영향력이 가장 큽니다.")

    return {
        "items": items,
        "insight": insight,
    }


def recommend_action(row: pd.Series) -> tuple[str, str]:
    if row["days_since_last_watch"] >= 30 and row["recent_watch_count"] <= 1:
        return "재활성화 캠페인", "즉시 대응"
    if row["completion_rate"] < 0.3 and row["watch_days"] < 10:
        return "가벼운 콘텐츠 추천", "우선 대응"
    if row["subscription_tenure_days"] >= 365 and row["churn_probability"] >= 0.75:
        return "충성고객 유지 혜택 제공", "즉시 대응"
    if row["plan_tier"] <= 1 and row["churn_probability"] >= 0.5:
        return "업그레이드 체험 제안", "관찰 필요"
    return "개인화 추천 메시지 발송", "일반 대응"


def describe_user_signal(row: pd.Series, top_features: list[str]) -> str:
    messages = []

    for feature in top_features[:3]:
        if feature == "days_since_last_watch":
            messages.append(f"미시청 {int(row['days_since_last_watch'])}일")
        elif feature == "recent_watch_count":
            messages.append(f"최근 시청 {float(row['recent_watch_count']):.0f}회")
        elif feature == "watch_days":
            messages.append(f"시청일수 {int(row['watch_days'])}일")
        elif feature == "completion_rate":
            messages.append(f"완주율 {float(row['completion_rate']) * 100:.1f}%")
        elif feature == "avg_progress":
            messages.append(f"평균 진행률 {float(row['avg_progress']):.1f}%")
        elif feature == "total_watch_time":
            messages.append(f"총 시청시간 {float(row['total_watch_time']):.1f}")
        elif feature == "subscription_tenure_days":
            messages.append(f"구독 {int(row['subscription_tenure_days'])}일")
        elif feature == "unique_movies":
            messages.append(f"콘텐츠 수 {int(row['unique_movies'])}개")

    if not messages:
        messages.append("행동 변화 감지")

    return " · ".join(messages)


def make_high_risk_users(
    result_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int = 8,
) -> list[dict]:
    top_features = importance_df["feature"].head(5).tolist()
    candidate_df = result_df.sort_values("churn_probability", ascending=False).head(top_n).copy()

    users = []
    for _, row in candidate_df.iterrows():
        user_id = row["user_id"] if "user_id" in result_df.columns else row.name
        action, priority = recommend_action(row)

        users.append(
            {
                "name": f"{user_id}",
                "risk": f"{float(row['risk_score']):.1f}%",
                "risk_label": row["risk_label"],
                "inactive_days": f"{int(row['days_since_last_watch'])}일",
                "watch_days": f"{int(row['watch_days'])}일",
                "recent_watch_count": f"{float(row['recent_watch_count']):.0f}회",
                "completion_rate": f"{float(row['completion_rate']) * 100:.1f}%",
                "plan_tier": f"{int(row['plan_tier'])}단계",
                "insight": describe_user_signal(row, top_features),
                "action": action,
                "priority": priority,
                "subline": (
                    f"미시청 {int(row['days_since_last_watch'])}일 · "
                    f"최근 시청 {float(row['recent_watch_count']):.0f}회 · "
                    f"완주율 {float(row['completion_rate']) * 100:.1f}%"
                ),
            }
        )

    return users


def make_campaign_recommendations(result_df: pd.DataFrame) -> list[dict]:
    total = max(len(result_df), 1)

    dormant_mask = (result_df["days_since_last_watch"] >= 30) & (result_df["churn_probability"] >= 0.75)
    loyal_risk_mask = (result_df["subscription_tenure_days"] >= 365) & (result_df["churn_probability"] >= 0.75)
    light_mask = (result_df["watch_days"] < 10) & (result_df["churn_probability"] >= 0.5)

    campaigns = [
        {
            "title": "재활성화 캠페인",
            "target_count": int(dormant_mask.sum()),
            "target_ratio": round(dormant_mask.mean() * 100, 1),
            "desc": "장기 미접속 고위험 고객 대상 복귀 유도",
        },
        {
            "title": "충성고객 유지 캠페인",
            "target_count": int(loyal_risk_mask.sum()),
            "target_ratio": round(loyal_risk_mask.mean() * 100, 1),
            "desc": "구독 기간이 길지만 이탈 위험이 높은 고객 보호",
        },
        {
            "title": "라이트 이용자 활성화",
            "target_count": int(light_mask.sum()),
            "target_ratio": round(light_mask.mean() * 100, 1),
            "desc": "사용량이 낮은 고객 대상 개인화 추천 강화",
        },
    ]

    for item in campaigns:
        item["share_of_total"] = f"전체의 {item['target_ratio']:.1f}%"

    return campaigns


def make_headline_insight(result_df: pd.DataFrame, importance_df: pd.DataFrame) -> str:
    high_risk_users = int((result_df["churn_probability"] >= 0.75).sum())
    high_risk_ratio = (result_df["churn_probability"] >= 0.75).mean() * 100

    top_2 = importance_df["feature_display"].head(2).tolist()
    joined = ", ".join(top_2)

    return (
        f"현재 고위험 고객은 {high_risk_users:,}명({high_risk_ratio:.1f}%)이며, "
        f"주요 이탈 신호는 {joined} 입니다."
    )


def build_analysis_payload(model: XGBClassifier, uploaded_file) -> dict:
    inference_df = prepare_inference_dataframe(uploaded_file)
    result_df = predict_with_model(model=model, inference_df=inference_df)
    importance_df = extract_feature_importance(model=model)

    uploaded_file_name = getattr(uploaded_file, "name", "uploaded_file")

    return {
        "headline_insight": make_headline_insight(result_df, importance_df),
        "data_meta": make_data_meta(result_df, uploaded_file_name),
        "kpi_data": make_kpi_data(result_df),
        "trend_data": make_trend_data(result_df),
        "risk_segments": make_risk_segments(result_df),
        "usage_data": make_usage_data(result_df),
        "profile_data": make_profile_data(result_df),
        "driver_data": make_driver_data(importance_df),
        "campaign_data": make_campaign_recommendations(result_df),
        "high_risk_users": make_high_risk_users(result_df, importance_df),
        "raw_result_df": result_df,
        "importance_df": importance_df,
    }