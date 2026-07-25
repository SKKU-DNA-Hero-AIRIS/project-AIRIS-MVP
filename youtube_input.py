"""Download a permitted YouTube video as a bounded local analysis input."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

MAX_DURATION_SECONDS = 180
ROOT = Path(__file__).resolve().parent
YTDLP_BINARY = ROOT / "tools/bin/yt-dlp_macos"


def run_ytdlp(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    executable = (
        str(YTDLP_BINARY)
        if YTDLP_BINARY.exists()
        else shutil.which("yt-dlp")
    )
    if not executable:
        raise FileNotFoundError(
            "최신 yt-dlp가 없습니다. Python 3.10+ 환경에서 requirements.txt를 설치하세요."
        )
    return subprocess.run(
        [executable, *arguments],
        check=True,
        text=True,
        capture_output=True,
    )


def download_youtube_video(url: str, destination: Path) -> tuple[Path, dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    metadata = run_ytdlp(
        ["--dump-single-json", "--no-playlist", "--skip-download", url]
    )
    info = json.loads(metadata.stdout)

    video_id = str(info["id"])
    duration = int(info.get("duration") or 0)
    if not duration or duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"영상 길이는 1초 이상 {MAX_DURATION_SECONDS}초 이하여야 합니다."
        )

    output_path = destination / f"{video_id}.mp4"
    if not output_path.exists():
        run_ytdlp(
            [
                "--no-playlist",
                "--format",
                "18/best[ext=mp4]/best",
                "--output",
                str(output_path),
                url,
            ]
        )

    return output_path, {
        "id": video_id,
        "title": info.get("title", video_id),
        "duration": duration,
        "webpage_url": info.get("webpage_url", url),
        "uploader": info.get("uploader"),
    }
