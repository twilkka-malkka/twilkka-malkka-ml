import streamlit as st


def init_state() -> None:
    defaults = {
        "current_view": "home",
        "uploaded_file": None,
        "uploaded_file_name": None,
        "is_sample_mode": False,
        "selected_model_name": "xgboost",
        "analysis_payload": None,
        "model_path": r"C:\Users\Playdata\Documents\workspace\twilkka-malkka-ml\01_notebooks\03_models\my_model\model.json",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_home() -> None:
    st.session_state.current_view = "home"


def go_dashboard() -> None:
    st.session_state.current_view = "dashboard"


def set_uploaded_file(uploaded_file) -> None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.uploaded_file_name = uploaded_file.name if uploaded_file is not None else None
    st.session_state.is_sample_mode = False
    st.session_state.analysis_payload = None


def clear_uploaded_file() -> None:
    st.session_state.uploaded_file = None
    st.session_state.uploaded_file_name = None
    st.session_state.is_sample_mode = False
    st.session_state.analysis_payload = None


def set_model_name(model_name: str) -> None:
    st.session_state.selected_model_name = model_name


def set_analysis_payload(payload: dict) -> None:
    st.session_state.analysis_payload = payload