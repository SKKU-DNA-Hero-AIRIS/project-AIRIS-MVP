"""Explain AIRIS fusion results through Groq or a local fallback."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"


def build_result_snapshot(context: Any, result: Any) -> dict[str, Any]:
    return {
        "overall_risk": result.overall_risk,
        "control_level": result.control_level,
        "confidence": result.confidence,
        "interpretation": result.interpretation,
        "reasons": list(result.reasons),
        "context": context.to_dict(),
    }


def build_system_prompt(snapshot: dict[str, Any]) -> str:
    return (
        "당신은 AIRIS 출입 안전 모니터링 도우미입니다. "
        "아래 실시간 측정 결과만 근거로 한국어로 짧고 명확하게 답하세요. "
        "측정값과 판단 근거를 먼저 설명하고, 확인되지 않은 진단은 하지 마세요. "
        "건강 관련 질문에는 이 결과가 의료 진단이 아니며 이상 수치가 지속되면 "
        "전문가의 확인이 필요하다고 안내하세요. 위험도와 제어 결과를 임의로 바꾸지 마세요.\n\n"
        f"AIRIS_RESULT:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}"
    )


def local_answer(question: str, snapshot: dict[str, Any]) -> str:
    context = snapshot["context"]
    health = context["health_context"]
    vision = context["vision_context"]
    environment = context["environment_context"]
    facility = context["facility_context"]
    lowered = question.lower()

    if any(word in lowered for word in ("체온", "열", "발열")):
        value = health["body_temperature"]
        note = "발열 가능성이 감지됐습니다." if value >= 38 else "현재 발열 기준 미만입니다."
        return (
            f"현재 체온은 {value:.1f}℃이며, {note} "
            "측정 결과가 지속되면 다시 측정하고 전문가에게 확인하세요."
        )
    if any(word in lowered for word in ("산소", "spo2", "포화도")):
        value = health["spo2"]
        note = "주의 기준 95% 미만입니다." if value < 95 else "주의 기준 이상입니다."
        return f"현재 산소포화도는 {value}%로 {note}"
    if any(word in lowered for word in ("심박", "맥박")):
        return f"현재 심박수는 {health['heart_rate']} bpm입니다."
    if any(word in lowered for word in ("사람", "인원", "혼잡", "마스크")):
        return (
            f"현재 {vision['person_count']}명이 감지됐고 혼잡도는 "
            f"{vision['crowding_level']}, 마스크 착용률은 "
            f"{vision['mask_rate'] * 100:.0f}%입니다."
        )
    if any(word in lowered for word in ("미세먼지", "pm2.5", "대기", "환경")):
        return (
            f"현재 PM2.5는 {environment['pm25']}㎍/㎥이고 "
            f"대기질은 {environment['air_quality']}입니다."
        )
    if any(word in lowered for word in ("시설", "병원", "학교")):
        return (
            f"현재 시설은 {facility['label']}이며 시설 민감도는 "
            f"{facility['sensitivity']}입니다."
        )

    reasons = " · ".join(snapshot["reasons"])
    return (
        f"종합 위험도는 {snapshot['overall_risk']}, 최종 제어는 "
        f"AIR GATE {snapshot['control_level']}입니다. 판단 근거는 {reasons}입니다."
    )


def ask_groq(
    api_key: str,
    messages: list[dict[str, str]],
    snapshot: dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> str:
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 350,
        "messages": [
            {"role": "system", "content": build_system_prompt(snapshot)},
            *messages[-8:],
        ],
    }
    request = Request(
        GROQ_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AIRIS-MVP/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"]).strip()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API 오류 ({error.code}): {detail[:160]}") from error
    except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("Groq API 응답을 받을 수 없습니다.") from error
