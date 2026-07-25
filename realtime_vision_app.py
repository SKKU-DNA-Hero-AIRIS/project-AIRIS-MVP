"""Realtime playback demo synchronized with AIRIS vision_context."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from external_context import (
    LOCATIONS,
    build_facility_context,
    fetch_external_context_by_coordinates,
    reverse_geocode,
)
from physiological_store import read_context, write_context
from vision_demo import analyze_video
from webcam_vision import LiveVisionState, WebcamVisionProcessor, load_webcam_models


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "outputs/firefly_cafe/annotated.mp4"
DEFAULT_RESULT = ROOT / "outputs/firefly_cafe/vision_context.json"


st.set_page_config(
    page_title="AIRIS Vision Context Live",
    page_icon="◉",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background:#081018; color:#e8f0f5; }
    .block-container { max-width:1480px; padding-top:2rem; }
    h1 { letter-spacing:-.04em; }
    [data-testid="stFileUploader"] {
      border:1px solid #243444; border-radius:12px; padding:.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("DNA-HERO · AIRIS MVP")
st.title("AIRIS 실시간 모니터링")

uploaded_file = None
upload_clicked = False

with st.sidebar:
    st.header("입력 방식")
    input_mode = st.radio(
        "분석할 영상을 선택하세요",
        ["영상 파일 업로드", "실시간 카메라", "생체 데이터", "외부 환경"],
        label_visibility="collapsed",
    )
    if input_mode == "영상 파일 업로드":
        st.divider()
        st.header("파일 업로드")
        uploaded_file = st.file_uploader(
            "영상 파일",
            type=["mp4", "mov", "avi", "mkv"],
            help="MP4, MOV, AVI, MKV 형식을 지원합니다.",
        )
        upload_clicked = st.button(
            "업로드 영상 분석",
            use_container_width=True,
            disabled=uploaded_file is None,
        )
    elif input_mode == "실시간 카메라":
        st.caption("카메라 영상은 저장하지 않습니다.")


@st.cache_resource(show_spinner="사람·마스크 탐지 모델을 불러오는 중…")
def get_webcam_models():
    return load_webcam_models()


if input_mode == "실시간 카메라":
    st.subheader("실시간 카메라")
    st.info("아래 `START`를 누르고 브라우저의 카메라 접근을 허용하세요.")

    if "live_vision_state" not in st.session_state:
        st.session_state["live_vision_state"] = LiveVisionState()
    live_state = st.session_state["live_vision_state"]
    person_model, mask_model = get_webcam_models()

    video_column, context_column = st.columns([1.55, 0.75], gap="large")
    with video_column:
        webcam_context = webrtc_streamer(
            key="airis-live-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            video_processor_factory=lambda: WebcamVisionProcessor(
                live_state,
                person_model,
                mask_model,
            ),
            async_processing=True,
        )

    if webcam_context.state.playing:
        st_autorefresh(interval=1000, key="webcam-json-refresh")

    live_payload = live_state.snapshot()
    vision_context = live_payload["vision_context"]
    with context_column:
        st.caption("CURRENT VISION CONTEXT")
        metric_left, metric_right = st.columns(2)
        metric_left.metric("PERSON COUNT", vision_context["person_count"])
        metric_right.metric(
            "MASK RATE",
            f"{round(vision_context['mask_rate'] * 100)}%",
        )
        st.metric("CROWDING LEVEL", vision_context["crowding_level"])
        st.json(live_payload)

    st.caption(
        "LOW: 0~3명 · MEDIUM: 4~7명 · HIGH: 8명 이상 · "
        "성능 최적화를 위해 사람은 매 6프레임, 마스크는 매 18프레임마다 분석합니다."
    )
    st.stop()

if input_mode == "생체 데이터":
    st.subheader("생체 데이터")

    physiological_role = st.radio(
        "사용 화면",
        ["데이터 전송", "실시간 모니터링"],
        horizontal=True,
    )

    if physiological_role == "데이터 전송":
        preset = st.selectbox("시나리오", ["정상", "운동 직후", "건강 이상"])
        presets = {
            "정상": (75, 98, 36.5, 20),
            "운동 직후": (125, 98, 36.8, 70),
            "건강 이상": (125, 94, 38.1, 85),
        }
        preset_hr, preset_spo2, preset_temp, preset_stress = presets[preset]

        with st.form("iphone-physiological-sender"):
            device_id = st.text_input("기기 이름", value="dongyeon_iphone")
            heart_rate = st.slider("심박수 (bpm)", 30, 200, preset_hr)
            spo2 = st.slider("산소포화도 (%)", 50, 100, preset_spo2)
            body_temperature = st.slider(
                "체온 (℃)",
                30.0,
                42.0,
                preset_temp,
                0.1,
            )
            stress_index = st.slider("스트레스 지수", 0, 100, preset_stress)
            send_clicked = st.form_submit_button(
                "전송",
                type="primary",
                use_container_width=True,
            )

        if send_clicked:
            sent_context = write_context(
                {
                    "device_id": device_id,
                    "heart_rate": heart_rate,
                    "spo2": spo2,
                    "body_temperature": body_temperature,
                    "stress_index": stress_index,
                }
            )
            st.success("전송 완료")
            st.json({"physiological_context": sent_context})
    else:
        st_autorefresh(interval=1000, key="physiological-monitor-refresh")
        physiological_context = read_context()
        timestamp = physiological_context["timestamp"]

        if timestamp is None:
            st.warning("수신 대기 중")
        else:
            st.success(f"{physiological_context['device_id']} · {timestamp}")

        physiological_metrics = st.columns(4)
        physiological_metrics[0].metric(
            "심박수",
            f"{physiological_context['heart_rate']} bpm",
        )
        physiological_metrics[1].metric(
            "산소포화도",
            f"{physiological_context['spo2']}%",
        )
        physiological_metrics[2].metric(
            "체온",
            f"{physiological_context['body_temperature']:.1f}℃",
        )
        physiological_metrics[3].metric(
            "스트레스",
            f"{physiological_context['stress_index']} / 100",
        )

        health_high = (
            physiological_context["spo2"] < 95
            or physiological_context["body_temperature"] >= 38.0
        )
        health_medium = (
            physiological_context["heart_rate"] >= 110
            or physiological_context["stress_index"] >= 65
        )
        if health_high:
            st.error("HEALTH RISK · HIGH")
        elif health_medium:
            st.warning("HEALTH RISK · MEDIUM")
        else:
            st.success("HEALTH RISK · LOW")

        standardized_context = {
            "timestamp": timestamp,
            "physiological_context": {
                key: physiological_context[key]
                for key in (
                    "heart_rate",
                    "spo2",
                    "body_temperature",
                    "stress_index",
                )
            },
        }
        st.subheader("JSON")
        st.json(standardized_context)

    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def get_external_context(
    latitude: float,
    longitude: float,
    location_name: str,
):
    return fetch_external_context_by_coordinates(
        latitude,
        longitude,
        location_name,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def get_place_info(latitude: float, longitude: float, cache_version: int = 2):
    del cache_version
    return reverse_geocode(latitude, longitude)


if input_mode == "외부 환경":
    st.subheader("외부 환경")
    default_latitude, default_longitude = LOCATIONS["성균관대 자연과학캠퍼스"]
    if "external_coordinates" not in st.session_state:
        st.session_state["external_coordinates"] = {
            "lat": default_latitude,
            "lng": default_longitude,
        }
    coordinates = st.session_state["external_coordinates"]

    location_map = folium.Map(
        location=[coordinates["lat"], coordinates["lng"]],
        zoom_start=16,
        control_scale=True,
    )
    folium.Marker(
        [coordinates["lat"], coordinates["lng"]],
        tooltip="선택 위치",
    ).add_to(location_map)
    map_result = st_folium(
        location_map,
        height=420,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="external-location-map",
    )
    clicked = map_result.get("last_clicked")
    if clicked:
        clicked_coordinates = {
            "lat": round(float(clicked["lat"]), 6),
            "lng": round(float(clicked["lng"]), 6),
        }
        if clicked_coordinates != coordinates:
            st.session_state["external_coordinates"] = clicked_coordinates
            coordinates = clicked_coordinates

    place_info = get_place_info(coordinates["lat"], coordinates["lng"], 2)
    inferred_facility = place_info["facility_type"]
    facility_options = ["병원", "어린이집", "학교", "공공기관", "사무실", "기타"]
    facility_type = st.selectbox(
        "시설 유형",
        facility_options,
        index=facility_options.index(inferred_facility),
        key=(
            f"facility-{coordinates['lat']:.5f}-"
            f"{coordinates['lng']:.5f}"
        ),
    )
    vulnerable_population = st.checkbox(
        "취약계층 이용 시설",
        value=facility_type in {"병원", "어린이집"},
        key=(
            f"vulnerable-{coordinates['lat']:.5f}-"
            f"{coordinates['lng']:.5f}"
        ),
    )
    refresh_external = st.button("새로고침", use_container_width=True)

    if refresh_external:
        get_external_context.clear()

    external_payload = get_external_context(
        coordinates["lat"],
        coordinates["lng"],
        place_info["name"],
    )
    environment_context = external_payload["environment_context"]
    facility_context = build_facility_context(
        facility_type,
        vulnerable_population,
    )

    external_metrics = st.columns(5)
    external_metrics[0].metric(
        "기온",
        f"{environment_context['temperature']:.1f}℃",
    )
    external_metrics[1].metric(
        "습도",
        f"{environment_context['humidity']}%",
    )
    external_metrics[2].metric(
        "PM2.5",
        f"{environment_context['pm25']:.1f}",
    )
    external_metrics[3].metric(
        "강수량",
        f"{environment_context['precipitation']:.1f} mm",
    )
    external_metrics[4].metric(
        "대기질",
        environment_context["air_quality"],
    )

    st.caption(
        f"{place_info['display_name']} · {external_payload['timestamp']} · "
        f"{external_payload['source']} · 지도 © OpenStreetMap"
    )
    st.subheader("JSON")
    st.json(
        {
            "environment_context": environment_context,
            "facility_context": facility_context,
        }
    )
    st.stop()


if upload_clicked and uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_id = hashlib.sha256(file_bytes).hexdigest()[:12]
        extension = Path(uploaded_file.name).suffix.lower() or ".mp4"
        upload_dir = ROOT / "inputs/uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        source_path = upload_dir / f"{file_id}{extension}"
        if not source_path.exists():
            source_path.write_bytes(file_bytes)

        result_dir = ROOT / "outputs/uploads" / file_id
        result_path = result_dir / "vision_context.json"
        annotated_path = result_dir / "annotated.mp4"
        with st.spinner("업로드 영상을 Vision Context로 분석하는 중…"):
            if not result_path.exists() or not annotated_path.exists():
                analyze_video(
                    source_path,
                    result_dir,
                    sample_every=6,
                )
            st.session_state["active_video"] = str(annotated_path)
            st.session_state["active_result"] = str(result_path)
            st.session_state["video_title"] = uploaded_file.name
        st.sidebar.success("업로드 영상 분석 완료")
    except Exception as error:
        st.sidebar.error(f"분석 실패: {error}")

active_video = Path(st.session_state.get("active_video", DEFAULT_VIDEO))
active_result = Path(st.session_state.get("active_result", DEFAULT_RESULT))

if not active_video.exists() or not active_result.exists():
    st.error("먼저 vision_demo.py로 영상을 분석해야 합니다.")
    st.stop()

st.subheader(st.session_state.get("video_title", "Firefly 카페 데모"))

payload = json.loads(active_result.read_text(encoding="utf-8"))
timeline = payload["timeline"]
video_base64 = base64.b64encode(active_video.read_bytes()).decode("ascii")

timeline_json = json.dumps(timeline, ensure_ascii=False)
summary_json = json.dumps(payload["vision_context"], ensure_ascii=False)

html = f"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:#081018; color:#e8f0f5;
    font-family:Inter, Pretendard, -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  .layout {{
    display:grid; grid-template-columns:minmax(0, 1.55fr) minmax(300px, .75fr);
    gap:18px; align-items:start;
  }}
  .video-shell, .context-shell {{
    border:1px solid #243444; border-radius:16px; overflow:hidden;
    background:#0d1822;
  }}
  video {{ display:block; width:100%; aspect-ratio:16/9; background:#000; }}
  .video-meta {{
    display:flex; justify-content:space-between; align-items:center;
    padding:12px 16px; color:#8ea1b0; font-size:13px;
  }}
  .live {{ color:#61d4c7; font-weight:700; letter-spacing:.08em; }}
  .context-shell {{ padding:18px; }}
  .label {{
    color:#708494; text-transform:uppercase; letter-spacing:.14em;
    font-size:11px; font-weight:700; margin-bottom:16px;
  }}
  .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
  .metric {{
    border:1px solid #243444; border-radius:12px; padding:14px;
    background:#101e29;
  }}
  .metric span {{ color:#8295a4; font-size:12px; display:block; margin-bottom:5px; }}
  .metric strong {{ font-size:27px; letter-spacing:-.03em; }}
  .risk {{
    margin-top:10px; padding:17px; border-radius:12px;
    background:#101e29; border:1px solid #243444;
  }}
  .risk strong {{ display:block; font-size:32px; margin-top:4px; }}
  .LOW {{ color:#61d4c7; }} .MEDIUM {{ color:#f4bd62; }} .HIGH {{ color:#ff7070; }}
  pre {{
    margin:14px 0 0; padding:14px; background:#071017; border-radius:10px;
    color:#9ed8d1; font:12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space:pre-wrap;
  }}
  .bar {{ height:4px; background:#1c2b37; border-radius:9px; margin-top:16px; }}
  .bar > div {{ height:100%; width:0; background:#61d4c7; border-radius:9px; }}
  details {{
    margin-top:16px; border:1px solid #243444; border-radius:14px;
    background:#0d1822; overflow:hidden;
  }}
  summary {{
    cursor:pointer; padding:16px 18px; font-weight:700;
    color:#dce7ee; user-select:none;
  }}
  details pre {{ margin:0 16px 16px; }}
  @media (max-width:850px) {{ .layout {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="layout">
  <section class="video-shell">
    <video id="video" controls autoplay muted playsinline>
      <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    <div class="video-meta">
      <span class="live">● LIVE CONTEXT</span>
      <span id="clock">00:00.00</span>
    </div>
  </section>
  <aside class="context-shell">
    <div class="label">Current vision context</div>
    <div class="metrics">
      <div class="metric"><span>PERSON COUNT</span><strong id="people">0</strong></div>
      <div class="metric"><span>MASK RATE</span><strong id="maskRate">0%</strong></div>
    </div>
    <div class="risk">
      <span style="color:#8295a4;font-size:12px">CROWDING LEVEL</span>
      <strong id="crowding" class="LOW">LOW</strong>
    </div>
    <div class="bar"><div id="progress"></div></div>
    <pre id="json"></pre>
  </aside>
</div>
<details open>
  <summary>영상 전체 요약 · 실시간 JSON</summary>
  <pre id="summaryJson"></pre>
</details>
<script>
  const timeline = {timeline_json};
  const finalSummary = {summary_json};
  const video = document.getElementById("video");

  function level(count) {{
    if (count >= 8) return "HIGH";
    if (count >= 4) return "MEDIUM";
    return "LOW";
  }}

  function sampleAt(time) {{
    let sample = timeline[0];
    for (const item of timeline) {{
      if (item.time_seconds <= time) sample = item;
      else break;
    }}
    return sample;
  }}

  function update() {{
    const sample = sampleAt(video.currentTime || 0);
    const people = sample.person_count || 0;
    const masks = Math.min(sample.masked_person_count || 0, people);
    const rate = people ? masks / people : 0;
    const crowding = level(people);

    document.getElementById("people").textContent = people;
    document.getElementById("maskRate").textContent = Math.round(rate * 100) + "%";
    const crowd = document.getElementById("crowding");
    crowd.textContent = crowding;
    crowd.className = crowding;
    document.getElementById("clock").textContent =
      "00:" + String(video.currentTime.toFixed(2)).padStart(5, "0");
    document.getElementById("progress").style.width =
      ((video.currentTime / (video.duration || 1)) * 100) + "%";
    const livePayload = {{
      timestamp_seconds: Number(video.currentTime.toFixed(2)),
      vision_context: {{
        person_count: people,
        masked_person_count: masks,
        mask_rate: Number(rate.toFixed(3)),
        crowding_level: crowding
      }}
    }};
    const renderedJson = JSON.stringify(livePayload, null, 2);
    document.getElementById("json").textContent = renderedJson;
    document.getElementById("summaryJson").textContent = renderedJson;
  }}

  video.addEventListener("timeupdate", update);
  video.addEventListener("seeked", update);
  video.addEventListener("loadedmetadata", update);
  setInterval(update, 100);
</script>
</body>
</html>
"""

components.html(html, height=900, scrolling=False)
