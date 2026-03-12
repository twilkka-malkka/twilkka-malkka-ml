import streamlit as st

from ..state.state import go_dashboard, set_uploaded_file


def render_home_view() -> None:
    st.markdown(
        """
        <div class="landing-wrap">
            <div class="landing-badge">OTT Churn Prediction</div>
            <div class="landing-title">🎬 튈까말까 OTT 고객 이탈 예측 서비스</div>
            <div class="landing-sub">
                고객 데이터를 업로드하면 저장된 모델로 이탈 위험도를 예측하고,
                실제 행동 특성과 함께 CRM 타깃 인사이트를 보여줍니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.5, 1])

    with center:
        uploaded_file = st.file_uploader(
            "고객 데이터 업로드",
            type=["csv", "xlsx"],
        )

        if uploaded_file is not None:
            set_uploaded_file(uploaded_file)
            st.success(f"업로드 완료: {uploaded_file.name}")

        if st.button(
            "분석 시작",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.uploaded_file is None,
        ):
            go_dashboard()
            st.rerun()