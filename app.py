import json

import streamlit as st

from airis import analyze_context, build_context
from airis.scenarios import (
    ENVIRONMENT_SCENARIOS,
    FACILITIES,
    HEALTH_SCENARIOS,
    VISION_SCENARIOS,
)


st.set_page_config(
    page_title="AIRIS — Context-Aware Air Gate",
    page_icon="◉",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');
    :root { --ink:#e8edf2; --muted:#93a0ad; --line:#26323d; --cyan:#61d4c7; }
    html, body, [class*="css"] { font-family:'IBM Plex Sans KR',sans-serif; }
    .stApp { background:#0b1015; color:var(--ink); }
    [data-testid="stSidebar"] { background:#10171e; border-right:1px solid var(--line); }
    .block-container { max-width:1400px; padding-top:2.3rem; }
    .eyebrow { color:#61d4c7; font-size:.74rem; letter-spacing:.18em; font-weight:700; }
    .hero-title { font-size:clamp(2rem,4vw,4rem); line-height:1.03; font-weight:700;
      letter-spacing:-.045em; margin:.4rem 0 .65rem; }
    .hero-sub { color:var(--muted); max-width:760px; font-size:1.03rem; }
    .panel { background:#111920; border:1px solid var(--line); border-radius:14px;
      padding:1.25rem 1.35rem; min-height:100%; }
    .panel-label { color:#788795; font-size:.72rem; letter-spacing:.12em;
      text-transform:uppercase; margin-bottom:.65rem; }
    .risk-low { color:#61d4c7; } .risk-medium { color:#f4bd62; }
    .risk-high { color:#ff6b6b; }
    .gate { border-radius:16px; padding:1.7rem; border:1px solid var(--line);
      background:linear-gradient(135deg,#121b23,#0e141a); }
    .gate-high { box-shadow:inset 5px 0 #ff6b6b; }
    .gate-medium { box-shadow:inset 5px 0 #f4bd62; }
    .gate-low { box-shadow:inset 5px 0 #61d4c7; }
    .gate-level { font-size:3.2rem; font-weight:700; letter-spacing:-.04em; }
    .reason { border-left:2px solid #394754; padding:.15rem 0 .15rem .8rem;
      color:#c6d0d9; margin:.58rem 0; }
    div[data-testid="stMetric"] { background:#111920; border:1px solid var(--line);
      padding:1rem; border-radius:12px; }
    div[data-testid="stMetricLabel"] { color:#83909c; }
    .stSelectbox label { color:#b5c0ca !important; }
    hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Demo controls")
    st.caption("입력값을 바꾸며 맥락 기반 판단을 확인하세요.")
    health_scenario = st.selectbox(
        "생체 시나리오", HEALTH_SCENARIOS, index=1, help="발표 기본값은 신체 활동입니다."
    )
    facility = st.selectbox(
        "시설 맥락", FACILITIES, index=0, help="사무실과 병원을 번갈아 비교하세요."
    )
    environment_scenario = st.selectbox(
        "환경 상태", ENVIRONMENT_SCENARIOS, index=0
    )
    vision_scenario = st.selectbox("공간 혼잡도", VISION_SCENARIOS, index=0)
    st.divider()
    st.markdown("**발표 핵심 비교**")
    st.caption("신체 활동 · 쾌적 · 여유를 고정하고 시설만 사무실 → 병원으로 바꿔보세요.")

context = build_context(
    health_scenario, environment_scenario, vision_scenario, facility
)
result = analyze_context(context)
level_class = result.control_level.lower()

st.markdown('<div class="eyebrow">DNA-HERO · AIRIS MVP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Context changes<br>the decision.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">동일한 센서값을 단순 임계값으로 처리하지 않습니다. '
    "사용자·환경·시설 맥락을 통합해 에어게이트 제어 수준과 이유를 설명합니다.</div>",
    unsafe_allow_html=True,
)
st.write("")

health = context.health_context
env = context.environment_context
vision = context.vision_context

metric_cols = st.columns(5)
metric_cols[0].metric("심박수", f"{health.heart_rate} bpm")
metric_cols[1].metric("SpO₂", f"{health.spo2}%")
metric_cols[2].metric("체온", f"{health.body_temperature:.1f}℃")
metric_cols[3].metric("PM2.5", f"{env.pm25} ㎍/㎥")
metric_cols[4].metric("현재 인원", f"{vision.person_count}명")

st.write("")
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown(
        f"""
        <div class="gate gate-{level_class}">
          <div class="panel-label">Air gate control</div>
          <div class="gate-level risk-{level_class}">{result.control_level}</div>
          <div style="color:#93a0ad;margin:.15rem 0 1rem">판단 신뢰도 {result.confidence}%</div>
          <div style="font-size:1.16rem;font-weight:600">{result.interpretation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 판단 근거")
    for reason in result.reasons:
        st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="panel-label">Risk fusion</div>', unsafe_allow_html=True)
    for label, value in (
        ("USER", result.user_risk),
        ("ENVIRONMENT", result.environment_risk),
        ("CROWDING", result.crowding_risk),
        ("FACILITY", result.facility_risk),
    ):
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:.68rem 0;'
            f'border-bottom:1px solid #26323d"><span style="color:#82909c">{label}</span>'
            f'<strong class="risk-{value.lower()}">{value}</strong></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
with st.expander("표준화된 Multimodal Context JSON 보기"):
    st.code(
        json.dumps(context.to_dict(), ensure_ascii=False, indent=2),
        language="json",
    )

st.caption(
    "Prototype mode · Synthetic health data + offline environment/vision mock data · "
    "실제 의료 진단 또는 안전 제어에 사용하지 마세요."
)
