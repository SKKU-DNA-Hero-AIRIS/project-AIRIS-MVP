"""Realtime playback demo synchronized with AIRIS vision_context."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from vision_demo import analyze_video
from youtube_input import download_youtube_video


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
st.title("Vision Context · Live Playback")
st.write(
    "영상 재생 위치와 동기화하여 현재 인원, 마스크 비율과 혼잡도를 표시합니다. "
    "재생바를 움직여도 데이터가 해당 시점으로 즉시 변경됩니다."
)

with st.sidebar:
    st.header("YouTube 입력")
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtu.be/...",
        help="3분 이하의 공개 영상만 지원합니다.",
    )
    has_permission = st.checkbox(
        "연구·다운로드 사용 권한을 확인했습니다",
        help="본인 영상 또는 사용 허가가 있는 영상만 입력하세요.",
    )
    analyze_clicked = st.button(
        "다운로드 후 분석",
        type="primary",
        use_container_width=True,
    )
    st.caption("첫 분석에는 모델 실행 때문에 몇 분이 걸릴 수 있습니다.")
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

if analyze_clicked:
    if not youtube_url.strip():
        st.sidebar.error("YouTube URL을 입력하세요.")
    elif not has_permission:
        st.sidebar.error("영상 사용 권한 확인이 필요합니다.")
    else:
        try:
            with st.spinner("YouTube 영상을 준비하고 Vision Context를 분석하는 중…"):
                source_path, video_info = download_youtube_video(
                    youtube_url.strip(),
                    ROOT / "inputs/youtube",
                )
                result_dir = ROOT / "outputs/youtube" / video_info["id"]
                result_path = result_dir / "vision_context.json"
                annotated_path = result_dir / "annotated.mp4"
                if not result_path.exists() or not annotated_path.exists():
                    analyze_video(
                        source_path,
                        result_dir,
                        sample_every=15,
                    )
                st.session_state["active_video"] = str(annotated_path)
                st.session_state["active_result"] = str(result_path)
                st.session_state["video_title"] = video_info["title"]
            st.sidebar.success("분석 완료")
        except Exception as error:
            st.sidebar.error(f"분석 실패: {error}")

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
