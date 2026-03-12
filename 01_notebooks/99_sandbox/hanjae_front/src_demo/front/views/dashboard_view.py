import streamlit as st

from ..state.state import clear_uploaded_file, go_home, set_analysis_payload
from ..ui.components import (
    render_campaign_recommendations,
    render_churn_drivers,
    render_data_meta,
    render_genre_chart,
    render_header,
    render_high_risk_users,
    render_kpi_card,
    render_ott_usage,
    render_risk_donut,
    render_section_heading,
    render_sticky_summary,
)
from ..viz.charts import make_trend_chart
from .analytics import build_analysis_payload, load_xgb_model


def _ensure_analysis_payload() -> dict:
    existing_payload = st.session_state.get("analysis_payload")
    if existing_payload is not None:
        return existing_payload

    model = load_xgb_model(st.session_state.model_path)
    payload = build_analysis_payload(
        model=model,
        uploaded_file=st.session_state.uploaded_file,
    )
    set_analysis_payload(payload)
    return payload


def render_dashboard_view() -> None:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    try:
        payload = _ensure_analysis_payload()
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        if st.button("← 메인으로 돌아가기"):
            go_home()
            st.rerun()
        return

    render_sticky_summary(payload["data_meta"], payload["driver_data"])

    top_left, top_right = st.columns([1.5, 1])

    with top_left:
        render_header(payload["headline_insight"])

    with top_right:
        render_data_meta(payload["data_meta"])

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("← 메인으로", use_container_width=True):
                go_home()
                st.rerun()

        with button_col2:
            if st.button("다시 업로드", use_container_width=True):
                clear_uploaded_file()
                go_home()
                st.rerun()

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    kpi_cols = st.columns(4)
    for col, item in zip(kpi_cols, payload["kpi_data"]):
        with col:
            render_kpi_card(
                title=item["title"],
                value=item["value"],
                subtext=item["subtext"],
                icon=item["icon"],
            )

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.7, 1])

    with left:
        render_section_heading(
            "📈 미시청 구간별 이탈 추세",
            "최근 미시청 일수 구간에 따라 평균 이탈확률과 최근 활동성이 어떻게 달라지는지 확인합니다.",
        )
        fig = make_trend_chart(payload["trend_data"])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        render_risk_donut(payload["risk_segments"])

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        render_ott_usage(payload["usage_data"])

    with col2:
        render_genre_chart(payload["profile_data"])

    with col3:
        render_churn_drivers(payload["driver_data"])

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    render_campaign_recommendations(payload["campaign_data"])

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    render_high_risk_users(payload["high_risk_users"])