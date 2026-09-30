"""Standalone dashboard for the personalized air gate optimizer.

    streamlit run personalized_gate_app.py --server.port 8504

Needs only numpy + streamlit, so it runs on machines where the camera stack
(ultralytics / av / webrtc) is not installed.
"""

from __future__ import annotations

import streamlit as st

from personalized_view import render_personalized_view


st.set_page_config(
    page_title="AIRIS Personalized Air Gate",
    page_icon="◉",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background:#081018; color:#e8f0f5; }
    .block-container { max-width:1480px; padding-top:2rem; }
    h1 { letter-spacing:-.04em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("DNA-HERO · AIRIS MVP")
st.title("AIRIS 맞춤형 에어게이트")

render_personalized_view()

st.caption(
    "Prototype mode · 축소 차수 물리 모델 + 합성 페르소나 · "
    "실제 의료 진단 또는 안전 제어에 사용하지 마세요."
)
