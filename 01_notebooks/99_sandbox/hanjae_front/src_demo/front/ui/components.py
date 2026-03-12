import streamlit as st
from textwrap import dedent

from ..viz.charts import make_genre_donut, make_risk_donut


def render_sticky_summary(data_meta: dict, driver_data: dict) -> None:
    top_driver = driver_data["items"][0]["label"] if driver_data["items"] else "-"
    st.markdown(
        dedent(
            f"""
            <div class="dashboard-summary-bar">
                총 사용자 <b>{data_meta['record_count']:,}</b>명 · 
                고위험 비중 <b>{data_meta['high_risk_ratio']:.1f}%</b> · 
                평균 예측 이탈확률 <b>{data_meta['avg_churn_prob']:.1f}%</b> · 
                주요 요인 <b>{top_driver}</b>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_header(headline_insight: str) -> None:
    st.markdown(
        dedent(
            f"""
            <div class="hero-wrap">
                <div class="hero-title">🎬 Netflix 고객 이탈 예측 서비스</div>
                <div class="hero-sub">실제 업로드 데이터와 모델 기반으로 이탈 위험과 CRM 타깃을 확인합니다.</div>
                <div class="hero-insight">
                    {headline_insight}
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_data_meta(data_meta: dict) -> None:
    st.markdown(
        dedent(
            f"""
            <div class="upload-card">
                <div class="upload-card-title">현재 분석 정보</div>
                <div class="upload-card-sub">파일명: {data_meta['file_name']}</div>
                <div class="upload-card-sub">레코드 수: {data_meta['record_count']:,}건</div>
                <div class="upload-card-sub">모델: {data_meta['model_name']}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, subtitle: str) -> None:
    st.markdown(
        dedent(
            f"""
            <div class="section-heading-wrap">
                <div class="section-title">{title}</div>
                <div class="section-sub">{subtitle}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, subtext: str, icon: str) -> None:
    st.markdown(
        dedent(
            f"""
            <div class="kpi-card">
                <div class="kpi-top">
                    <div class="kpi-icon">{icon}</div>
                </div>
                <div class="kpi-label">{title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-subtext">{subtext}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_risk_donut(risk_segments: dict) -> None:
    render_section_heading("⚠️ 위험도 분포", "예측 확률 기준 사용자 세그먼트 분포")

    fig = make_risk_donut(risk_segments)
    st.plotly_chart(fig, use_container_width=True)

    total = max(sum(risk_segments["values"]), 1)
    colors = ["#E50914", "#F97316", "#FACC15", "#22C55E"]

    for idx, (label, value) in enumerate(zip(risk_segments["labels"], risk_segments["values"])):
        pct = value / total * 100
        st.markdown(
            dedent(
                f"""
                <div class="legend-item">
                    <div class="legend-left">
                        <div class="legend-dot" style="background:{colors[idx]};"></div>
                        <div class="legend-label">{label}</div>
                    </div>
                    <div class="legend-right">
                        <span class="legend-value">{value:,}명</span>
                        <span class="legend-pct">({pct:.1f}%)</span>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    high_risk_pct = risk_segments["values"][0] / total * 100
    st.markdown(
        dedent(
            f"""
            <div class="alert-box">
                전체의 <b>{high_risk_pct:.1f}%</b>가 즉시 관리가 필요한 고위험군입니다.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_ott_usage(usage_data: dict) -> None:
    render_section_heading("▶️ 이용 빈도 분포", "시청일수를 기준으로 나눈 사용자 이용 유형")

    for item in usage_data["items"]:
        st.markdown(
            dedent(
                f"""
                <div class="metric-row">
                    <div class="metric-row-top">
                        <div class="metric-label">{item["label"]}</div>
                        <div class="metric-value">{item["value"]:.1f}%</div>
                    </div>
                    <div class="metric-desc">{item["desc"]}</div>
                    <div class="bar-track">
                        <div class="bar-fill-blue" style="width:{item["value"]}%;"></div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        dedent(
            f"""
            <div class="alert-box">
                {usage_data["insight"]}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_genre_chart(profile_data: dict) -> None:
    render_section_heading("🎞 사용자 시청 프로필", "행동 기반으로 나눈 주요 사용자 유형 비중")

    fig = make_genre_donut(profile_data["items"])
    st.plotly_chart(fig, use_container_width=True)

    colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444"]

    for idx, item in enumerate(profile_data["items"]):
        color = colors[idx % len(colors)]
        st.markdown(
            dedent(
                f"""
                <div class="legend-item">
                    <div class="legend-left">
                        <div class="legend-dot" style="background:{color};"></div>
                        <div class="legend-label">{item["label"]}</div>
                    </div>
                    <div class="legend-right">
                        <span class="legend-value">{item["value"]:.1f}%</span>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        dedent(
            f"""
            <div class="alert-box">
                {profile_data["insight"]}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_churn_drivers(driver_data: dict) -> None:
    render_section_heading("📉 이탈 예측 주요 요인", "선택한 모델의 피처 중요도 기준 핵심 요인")

    for item in driver_data["items"]:
        st.markdown(
            dedent(
                f"""
                <div class="metric-row">
                    <div class="metric-row-top">
                        <div class="metric-label">{item["label"]}</div>
                        <div class="metric-value">{item["value"]:.1f}%</div>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill-red" style="width:{item["value"]}%;"></div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        dedent(
            f"""
            <div class="alert-box">
                {driver_data["insight"]}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_campaign_recommendations(campaign_data: list[dict]) -> None:
    render_section_heading(
        "🎯 추천 캠페인",
        "현재 업로드 데이터 기준으로 바로 실행할 수 있는 CRM 액션 제안입니다.",
    )

    cols = st.columns(len(campaign_data))
    for col, item in zip(cols, campaign_data):
        with col:
            st.markdown(
                dedent(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{item["title"]}</div>
                        <div class="kpi-value">{item["target_count"]:,}명</div>
                        <div class="kpi-subtext">{item["share_of_total"]}</div>
                        <div class="metric-desc" style="margin-top:10px;">{item["desc"]}</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

def render_high_risk_users(users: list[dict]) -> None:
    render_section_heading(
        "🚨 높은 이탈 위험 사용자",
        "위험도와 행동 신호를 기준으로 우선 CRM 대상 고객을 확인합니다.",
    )

    for user in users:
        user_html = f"""
<div class="user-card">
    <div class="user-head">
        <div class="user-head-left">
            <div class="user-name">🔴 고객 {user["name"]}</div>
            <div class="user-subline">{user["subline"]}</div>
        </div>
        <div class="user-pill">{user["risk_label"]} · 위험도 {user["risk"]}</div>
    </div>
    <div class="campaign-chip-wrap">
        <span class="campaign-chip">{user["action"]}</span>
        <span class="campaign-chip secondary-chip">{user["priority"]}</span>
    </div>
    <div class="user-grid">
        <div class="user-meta highlight-meta">
            <div class="user-meta-label">위험도</div>
            <div class="user-meta-value">{user["risk"]}</div>
        </div>
        <div class="user-meta highlight-meta">
            <div class="user-meta-label">미시청 일수</div>
            <div class="user-meta-value">{user["inactive_days"]}</div>
        </div>
        <div class="user-meta">
            <div class="user-meta-label">시청 일수</div>
            <div class="user-meta-value">{user["watch_days"]}</div>
        </div>
        <div class="user-meta">
            <div class="user-meta-label">최근 시청 횟수</div>
            <div class="user-meta-value">{user["recent_watch_count"]}</div>
        </div>
        <div class="user-meta">
            <div class="user-meta-label">완주율</div>
            <div class="user-meta-value">{user["completion_rate"]}</div>
        </div>
        <div class="user-meta">
            <div class="user-meta-label">요금제</div>
            <div class="user-meta-value">{user["plan_tier"]}</div>
        </div>
    </div>
    <div class="alert-box" style="margin-top:16px;">
        핵심 행동 신호: <b>{user["insight"]}</b>
    </div>
</div>
"""
        st.markdown(user_html, unsafe_allow_html=True)
