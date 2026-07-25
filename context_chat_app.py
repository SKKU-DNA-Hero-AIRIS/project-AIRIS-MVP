"""Experimental chat UI for discussing the latest AIRIS fusion result."""

from __future__ import annotations

import os

import streamlit as st

from env_config import read_env_value, save_env_value
from external_context import read_latest_external_context
from integrated_context import analyze_live_context
from live_context_store import read_facility_context, read_vision_context
from physiological_store import read_context
from risk_chat import DEFAULT_MODEL, ask_groq, build_result_snapshot, local_answer


st.set_page_config(page_title="AIRIS Context Chat", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background:#081018; color:#e8f0f5; }
    .block-container { max-width:1180px; padding-top:2rem; }
    .risk-card {
      border:1px solid #314457; border-radius:18px; padding:24px 28px;
      background:linear-gradient(135deg,#101d29,#0b151f); margin:.5rem 0 1.2rem;
    }
    .risk-card.high { border-color:#ef5350; }
    .risk-card.medium { border-color:#ffb547; }
    .risk-card.low { border-color:#37c98b; }
    .risk-label { color:#91a4b5; font-size:12px; font-weight:700; letter-spacing:.14em; }
    .risk-value { font-size:46px; line-height:1.1; font-weight:800; margin-top:4px; }
    .risk-value.high { color:#ff6562; }
    .risk-value.medium { color:#ffc263; }
    .risk-value.low { color:#48d99d; }
    .risk-detail { color:#becbd5; margin-top:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def secret_or_env(name: str) -> str:
    try:
        secret = str(st.secrets.get(name, ""))
    except (FileNotFoundError, KeyError):
        secret = ""
    return secret or os.getenv(name, "") or read_env_value(name)


with st.sidebar:
    st.header("실험 설정")
    groq_key = st.text_input(
        "Groq API Key",
        value=secret_or_env("GROQ_API_KEY"),
        type="password",
        help="키가 없으면 기본 응답 모드로 작동합니다.",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    if st.button("키 저장", use_container_width=True):
        try:
            save_env_value("GROQ_API_KEY", groq_key)
            st.success("로컬 .env에 저장했습니다.")
        except ValueError as error:
            st.error(str(error))
    if st.button("대화 초기화", use_container_width=True):
        st.session_state["risk_chat_messages"] = []
        st.rerun()

latest_vision = read_vision_context()
latest_health = read_context()
latest_external = read_latest_external_context()
latest_facility = read_facility_context()
context, decision = analyze_live_context(
    latest_vision,
    latest_health,
    latest_external["environment_context"],
    latest_facility,
)
snapshot = build_result_snapshot(context, decision)

st.caption("AIRIS · EXPERIMENT")
st.title("Context 결과 대화")
risk_class = decision.overall_risk.lower()
reasons = " · ".join(decision.reasons)
st.markdown(
    f"""
    <div class="risk-card {risk_class}">
      <div class="risk-label">OVERALL RISK</div>
      <div class="risk-value {risk_class}">{decision.overall_risk}</div>
      <div class="risk-detail">
        최종 제어 <b>AIR GATE {decision.control_level}</b>
        &nbsp;·&nbsp; 신뢰도 <b>{decision.confidence}%</b><br>
        {reasons}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_columns[0].metric(
    "체온",
    f"{latest_health['body_temperature']:.1f}℃",
    f"심박 {latest_health['heart_rate']} bpm",
    delta_color="off",
)
metric_columns[1].metric(
    "산소포화도",
    f"{latest_health['spo2']}%",
    f"스트레스 {latest_health['stress_index']}",
    delta_color="off",
)
metric_columns[2].metric(
    "혼잡도",
    latest_vision["crowding_level"],
    f"{latest_vision['person_count']}명",
    delta_color="off",
)
metric_columns[3].metric(
    "대기질",
    latest_external["environment_context"]["air_quality"],
    f"PM2.5 {latest_external['environment_context']['pm25']}",
    delta_color="off",
)

st.divider()
if "risk_chat_messages" not in st.session_state:
    st.session_state["risk_chat_messages"] = [
        {
            "role": "assistant",
            "content": (
                f"현재 종합 위험도는 {decision.overall_risk}입니다. "
                "측정 결과에 대해 질문해 주세요."
            ),
        }
    ]

for message in st.session_state["risk_chat_messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("예: 왜 위험도가 높나요? 체온은 괜찮나요?")
if question:
    st.session_state["risk_chat_messages"].append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if groq_key.strip():
            with st.spinner("분석 중…"):
                try:
                    answer = ask_groq(
                        groq_key.strip(),
                        st.session_state["risk_chat_messages"],
                        snapshot,
                        model.strip() or DEFAULT_MODEL,
                    )
                except RuntimeError as error:
                    st.warning(str(error))
                    answer = local_answer(question, snapshot)
        else:
            answer = local_answer(question, snapshot)
        st.write(answer)
    st.session_state["risk_chat_messages"].append(
        {"role": "assistant", "content": answer}
    )
