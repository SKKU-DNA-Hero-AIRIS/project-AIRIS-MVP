"""Streamlit view: fixed actuation vs. the plan fitted to one user's body.

Shared by ``realtime_vision_app.py`` (the "맞춤 최적화" mode) and the standalone
``personalized_gate_app.py``. It imports nothing camera-related, so it runs
without the vision stack installed.
"""

from __future__ import annotations

import json
from html import escape

import streamlit as st

from airis.body_profile import (
    AGE_GROUPS,
    CLOTHING_TYPES,
    COMFORT_LEVELS,
    MOBILITY_LEVELS,
    POSTURES,
    UserProfile,
    UserTraits,
    body_from_measurements,
    load_user_profiles,
)
from airis.gate_physics import DEFAULT_GATE, ComfortLimits, SimulationResult
from airis.models import LEVELS
from airis.personalization import PersonalizedPlan, plan_personalized
from external_context import read_latest_external_context
from integrated_context import analyze_personalized_context
from live_context_store import (
    read_facility_context,
    read_user_profile,
    read_vision_context,
)
from physiological_store import read_context


LIVE_LEVEL = "통합 판단 연동"
STATUS_TEXT = {
    "MET": ("목표 충족", "ok"),
    "WITHIN_TOLERANCE": ("허용 오차 내 미달", "warn"),
    "SHORTFALL": ("목표 미달 · 관리자 확인", "bad"),
}
LABELS = {
    "CHILD": "어린이", "ADULT": "성인", "SENIOR": "고령",
    "NORMAL": "보통", "SLOW": "느림", "ASSISTED": "보조기구",
    "SENSITIVE": "민감", "LIGHT": "가벼움", "BULKY": "부피 큼",
    "STANDING": "서 있음", "SEATED": "착석(휠체어)",
}

# Ink and surface tokens match the dashboard's dark theme. Jet strength is one
# hue stepped by opacity; status is always paired with a text label.
INK, MUTED, FAINT = "#e8f0f5", "#91a4b5", "#5b6d7d"
JET, BODY_FILL, BODY_LINE, POST = "#61d4c7", "#22303d", "#4b5d6e", "#1b2733"
ALERT = "#ff6562"

STYLE = """
<style>
.pg-card { border:1px solid #2a3b4d; border-radius:16px; padding:18px 22px;
  background:linear-gradient(135deg,#101d29 0%,#0b151f 100%); }
.pg-card.ok { border-color:#37c98b; } .pg-card.warn { border-color:#ffb547; }
.pg-card.bad { border-color:#ef5350; }
.pg-eyebrow { color:#91a4b5; font-size:12px; font-weight:700; letter-spacing:.14em; }
.pg-title { font-size:26px; font-weight:800; margin-top:2px; color:#e8f0f5; }
.pg-meta { color:#c0ccd6; font-size:15px; margin-top:8px; }
.pg-reason { border-left:2px solid #394754; padding:.12rem 0 .12rem .8rem;
  color:#c6d0d9; margin:.5rem 0; }
.pg-figure { border:1px solid #243444; border-radius:14px; padding:8px 6px 2px;
  background:#0b151f; }
.pg-note { color:#91a4b5; font-size:13px; }
</style>
"""


def _label(value: str) -> str:
    return LABELS.get(value, value)


def _wh(energy_wh: float) -> str:
    """LOW-level passages use a few mWh; keep them from rounding to 0.00."""
    return f"{energy_wh:.3f}" if energy_wh < 0.1 else f"{energy_wh:.2f}"


def _manual_profile() -> UserProfile:
    columns = st.columns(4)
    posture = columns[0].selectbox("자세", POSTURES, format_func=_label)
    height = columns[1].number_input(
        "머리 끝 높이 (m)", 0.90, 2.10, 1.30 if posture == "SEATED" else 1.70, 0.01
    )
    shoulder = columns[2].number_input("어깨폭 (m)", 0.20, 0.65, 0.42, 0.01)
    mass = columns[3].number_input(
        "체중 (kg, 0 = 체형으로 추정)", 0.0, 160.0, 0.0, 1.0
    )
    columns = st.columns(5)
    speed = columns[0].number_input("보행속도 (m/s)", 0.3, 2.0, 1.2, 0.1)
    age_group = columns[1].selectbox("연령대", AGE_GROUPS, index=1, format_func=_label)
    mobility = columns[2].selectbox("이동성", MOBILITY_LEVELS, format_func=_label)
    sensitivity = columns[3].selectbox("민감도", COMFORT_LEVELS, format_func=_label)
    clothing = columns[4].selectbox(
        "의류", CLOTHING_TYPES, index=1, format_func=_label
    )
    return UserProfile(
        body=body_from_measurements(
            height, shoulder, mass_kg=mass or None, posture=posture
        ),
        traits=UserTraits(
            walking_speed_mps=speed,
            age_group=age_group,
            mobility=mobility,
            comfort_sensitivity=sensitivity,
            clothing=clothing,
        ),
        profile_id="MANUAL",
        label="직접 입력 사용자",
    )


def _select_profile() -> UserProfile:
    personas = load_user_profiles()
    source = st.radio(
        "사용자 프로파일",
        ["페르소나", "직접 입력", "실시간 측정"],
        horizontal=True,
    )
    if source == "페르소나":
        return st.selectbox(
            "페르소나",
            personas,
            format_func=lambda profile: f"{profile.profile_id} · {profile.label}",
        )
    if source == "직접 입력":
        return _manual_profile()

    measured = read_user_profile()
    if "body_profile" not in measured:
        st.info(
            "아직 측정된 체형이 없습니다. `보행 분석` 모드에서 전신이 보이게 통과하면 "
            "여기에 반영됩니다. 지금은 표준 체형 성인(P1)으로 계산합니다."
        )
        return personas[0]
    return UserProfile.from_dict(measured)


def _plan(profile: UserProfile, level_choice: str, crowding: str) -> PersonalizedPlan:
    if level_choice != LIVE_LEVEL:
        return plan_personalized(profile, level_choice, crowding_level=crowding)
    _, _, plan = analyze_personalized_context(
        read_vision_context(),
        read_context(),
        read_latest_external_context()["environment_context"],
        read_facility_context(),
        profile.to_dict(),
    )
    return plan


def gate_svg(
    result: SimulationResult,
    profile: UserProfile,
    limits: ComfortLimits,
    title: str,
) -> str:
    """Front view of the gate: nozzle rows, jets and the user's three body zones."""
    scale, floor, center = 150.0, 350.0, 170.0
    half_gate = DEFAULT_GATE.passage_width_m / 2.0 * scale
    left, right = center - half_gate, center + half_gate
    top = floor - 2.1 * scale
    face_over = result.face_velocity_mps > limits.face_velocity_mps + 0.05

    parts = [
        f'<svg viewBox="0 0 340 376" role="img" aria-label="{escape(title)}" '
        'xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%;height:auto;font-family:inherit">',
        f'<text x="{center}" y="16" fill="{INK}" font-size="13" font-weight="700" '
        f'text-anchor="middle">{escape(title)}</text>',
        f'<text x="{left - 14}" y="30" fill="{FAINT}" font-size="9" '
        'text-anchor="end">노즐 m/s</text>',
        f'<text x="{right + 14}" y="30" fill="{FAINT}" font-size="9">표면 m/s</text>',
        f'<line x1="30" y1="{floor}" x2="310" y2="{floor}" stroke="{BODY_LINE}" />',
    ]
    for x in (left - 8, right):
        parts.append(
            f'<rect x="{x}" y="{top}" width="8" height="{floor - top}" rx="3" '
            f'fill="{POST}" />'
        )

    for segment in profile.body.segments():
        width = segment.half_width_m * 2.0 * scale
        y = floor - segment.z_high_m * scale
        height = segment.span_m * scale
        outline = ALERT if segment.zone == "HEAD" and face_over else BODY_LINE
        radius = min(width, height) / 2.0 if segment.zone == "HEAD" else 8
        parts.append(
            f'<rect x="{center - width / 2:.1f}" y="{y + 1:.1f}" width="{width:.1f}" '
            f'height="{height - 2:.1f}" rx="{radius:.1f}" fill="{BODY_FILL}" '
            f'stroke="{outline}" stroke-width="{2 if outline == ALERT else 1}" />'
        )

    for row in result.rows:
        y = floor - row.height_m * scale
        velocity = row.nozzle_velocity_mps
        tip = (
            f"높이 {row.height_m:.2f} m · 노즐 {velocity:.1f} m/s"
            + (
                f" · 표면 {row.surface_velocity_mps:.1f} m/s ({row.zone})"
                if row.zone
                else " · 닿는 신체 없음"
            )
        )
        parts.append(f"<g><title>{escape(tip)}</title>")
        if velocity <= 0:
            for x in (left - 4, right + 4):
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="3" fill="none" stroke="{FAINT}" />'
                )
            parts.append(
                f'<text x="{left - 14}" y="{y + 3.5}" fill="{FAINT}" font-size="10" '
                'text-anchor="end">off</text>'
            )
        else:
            strength = velocity / DEFAULT_GATE.max_jet_velocity_mps
            reach = (
                (row.standoff_m or 0.0) * scale if row.zone else half_gate - 6
            )
            stroke = 1.5 + 4.0 * strength
            opacity = 0.25 + 0.75 * strength
            for x, direction in ((left, 1), (right, -1)):
                parts.append(
                    f'<line x1="{x}" y1="{y}" x2="{x + direction * reach:.1f}" '
                    f'y2="{y}" stroke="{JET}" stroke-width="{stroke:.1f}" '
                    f'stroke-linecap="round" opacity="{opacity:.2f}" />'
                )
            parts.append(
                f'<text x="{left - 14}" y="{y + 3.5}" fill="{MUTED}" font-size="10" '
                f'text-anchor="end">{velocity:.1f}</text>'
            )
            surface = f"{row.surface_velocity_mps:.1f}" if row.zone else "허공"
            parts.append(
                f'<text x="{right + 14}" y="{y + 3.5}" fill="{MUTED}" '
                f'font-size="10">{surface}</text>'
            )
        parts.append("</g>")

    parts.append(
        f'<text x="{center}" y="{floor + 18}" fill="{MUTED}" font-size="10" '
        f'text-anchor="middle">통로 폭 {DEFAULT_GATE.passage_width_m:.2f} m · '
        f'정면도</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _figure(
    column, result: SimulationResult, plan: PersonalizedPlan, title: str, notes: list[str]
) -> None:
    with column:
        st.markdown(
            '<div class="pg-figure">'
            + gate_svg(result, plan.profile, plan.limits, title)
            + "</div>",
            unsafe_allow_html=True,
        )
        for note in notes:
            st.markdown(f'<div class="pg-note">{escape(note)}</div>', unsafe_allow_html=True)


def _comparison_rows(level: str, crowding: str) -> list[dict[str, str]]:
    rows = []
    for profile in load_user_profiles():
        plan = plan_personalized(profile, level, crowding_level=crowding)
        rows.append(
            {
                "사용자": f"{profile.profile_id} · {profile.label}",
                "목표": f"{plan.target_removal:.1%}",
                "고정 제거율": f"{plan.baseline.removal_efficiency:.1%}",
                "고정 안전": "한계 초과" if plan.baseline_violations else "이내",
                "맞춤 제거율": f"{plan.personalized.removal_efficiency:.1%}",
                "맞춤 얼굴 m/s": f"{plan.personalized.face_velocity_mps:.1f}",
                "에너지 Wh (고정→맞춤)": (
                    f"{_wh(plan.baseline.energy_wh)} → {_wh(plan.personalized.energy_wh)}"
                ),
                "통과 s (고정→맞춤)": (
                    f"{plan.baseline.passage_time_s:.1f} → "
                    f"{plan.personalized.passage_time_s:.1f}"
                ),
                "상태": STATUS_TEXT[plan.target_status][0],
            }
        )
    return rows


def render_personalized_view() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    st.subheader("체형·사용자 특성 기반 맞춤 최적화")
    st.caption(
        "맥락 판단이 정한 제어 수준을, 지금 게이트에 선 사람의 체형과 특성에 맞는 "
        "노즐 풍속·정지 시간으로 바꿉니다. 물리 모델은 축소 차수 모델이며 CFD·실측이 아닙니다."
    )

    profile = _select_profile()
    columns = st.columns([1.4, 1])
    level_choice = columns[0].radio(
        "제어 수준", [LIVE_LEVEL, *LEVELS], index=3, horizontal=True
    )
    crowding = columns[1].radio(
        "혼잡도 (대기 비용)",
        LEVELS,
        horizontal=True,
        disabled=level_choice == LIVE_LEVEL,
        help="혼잡할수록 정지 시간을 줄이고 풍속으로 보상합니다.",
    )

    plan = _plan(profile, level_choice, crowding)
    fixed, fitted = plan.baseline, plan.personalized
    status_text, status_class = STATUS_TEXT[plan.target_status]
    body = profile.body

    st.markdown(
        f"""
        <div class="pg-card {status_class}">
          <div class="pg-eyebrow">PERSONALIZED AIR GATE · {plan.control_level}</div>
          <div class="pg-title">{escape(profile.label)} — {status_text}</div>
          <div class="pg-meta">
            키 <b>{body.height_m:.2f} m</b> · 어깨폭 <b>{body.shoulder_width_m:.2f} m</b>
            · 체중 <b>{body.mass_kg:.0f} kg</b> ({body.build})
            · 보행 <b>{profile.traits.walking_speed_mps:.1f} m/s</b>
            &nbsp;|&nbsp; 한계: 얼굴 ≤ <b>{plan.limits.face_velocity_mps:.0f} m/s</b>,
            측면 하중 ≤ <b>{plan.limits.lateral_load_ratio:.1%}</b>
            ({escape(plan.limits.basis)})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    metrics = st.columns(5)
    metrics[0].metric(
        "제거 효율",
        f"{fitted.removal_efficiency:.1%}",
        f"{(fitted.removal_efficiency - fixed.removal_efficiency) * 100:+.1f}%p vs 고정",
        help=f"목표 {plan.target_removal:.1%} (고정 제어가 설계 기준 성인에게 내는 성능)",
    )
    metrics[1].metric(
        "얼굴 도달 풍속",
        f"{fitted.face_velocity_mps:.1f} m/s",
        f"{fitted.face_velocity_mps - fixed.face_velocity_mps:+.1f} m/s vs 고정",
        delta_color="inverse",
    )
    metrics[2].metric(
        "측면 하중비",
        f"{fitted.lateral_load_ratio:.2%}",
        f"{(fitted.lateral_load_ratio - fixed.lateral_load_ratio) * 100:+.2f}%p vs 고정",
        delta_color="inverse",
    )
    metrics[3].metric(
        "통과당 에너지",
        f"{_wh(fitted.energy_wh)} Wh",
        f"{-plan.energy_saving_pct:+.0f}% vs 고정",
        delta_color="inverse",
    )
    metrics[4].metric(
        "통과 시간",
        f"{fitted.passage_time_s:.1f} s",
        f"{fitted.passage_time_s - fixed.passage_time_s:+.1f} s vs 고정",
        delta_color="inverse",
    )

    st.write("")
    left, right = st.columns(2, gap="large")
    fixed_notes = [f"제거율 {fixed.removal_efficiency:.1%} · {_wh(fixed.energy_wh)} Wh"]
    fixed_notes += [f"⚠ 한계 초과 — {violation}" for violation in plan.baseline_violations]
    _figure(left, fixed, plan, "고정 제어 (체형 무관)", fixed_notes)
    _figure(
        right,
        fitted,
        plan,
        "맞춤 제어 (이 사용자)",
        [
            f"제거율 {fitted.removal_efficiency:.1%} · {_wh(fitted.energy_wh)} Wh",
            f"머리 {plan.setpoints.head_velocity_mps:.1f} / 상체 "
            f"{plan.setpoints.upper_velocity_mps:.1f} / 하체 "
            f"{plan.setpoints.lower_velocity_mps:.1f} m/s · 정지 "
            f"{plan.setpoints.hold_seconds:.1f} s",
        ],
    )
    st.caption(
        "선 굵기·진하기 = 노즐 풍속. 왼쪽 숫자는 노즐 출구, 오른쪽 숫자는 몸 표면 도달 풍속. "
        "각 열에 마우스를 올리면 상세 값이 표시됩니다."
    )

    st.markdown("#### 왜 이렇게 바꿨나")
    for reason in plan.reasons:
        st.markdown(f'<div class="pg-reason">{escape(reason)}</div>', unsafe_allow_html=True)
    st.caption(
        f"탐색 방식 {plan.method} · 물리 시뮬레이터 호출 {plan.simulator_calls}회"
    )

    with st.expander("노즐 열별 설정 표"):
        st.dataframe(
            [
                {
                    "높이 (m)": row.height_m,
                    "신체 구역": row.zone or "—",
                    "고정 노즐 m/s": fixed_row.nozzle_velocity_mps,
                    "맞춤 노즐 m/s": row.nozzle_velocity_mps,
                    "맞춤 표면 m/s": row.surface_velocity_mps,
                }
                for row, fixed_row in zip(reversed(fitted.rows), reversed(fixed.rows))
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown(f"#### 같은 제어 수준({plan.control_level}), 다른 몸")
    st.dataframe(
        _comparison_rows(plan.control_level, crowding),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Personalized Plan JSON"):
        st.code(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), language="json")
