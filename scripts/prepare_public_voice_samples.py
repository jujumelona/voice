from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "input_test_public_voice"
DEFAULT_FFMPEG = ROOT / "bin" / "ffmpeg" / "ffmpeg.exe"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VoiceBridgeTest/1.0"

SAMPLE_SPECS = [
    {
        "id": "prime_minister_gillard_2011",
        "output_filename": "prime_minister_gillard_2011.wav",
        "original_filename": "Prime Minister Gillard of Australia at News Conference with President Obama.flac",
        "download_url": (
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Prime%20Minister%20Gillard%20of%20Australia%20at%20News%20Conference%20with%20President%20Obama.flac"
        ),
        "source_page": (
            "https://commons.wikimedia.org/wiki/"
            "File:Prime_Minister_Gillard_of_Australia_at_News_Conference_with_President_Obama.flac"
        ),
        "license_name": "Public domain (work of the U.S. federal government)",
        "license_url": "https://en.wikipedia.org/wiki/Copyright_status_of_work_by_the_U.S._government",
        "author": "The White House",
        "speaker": "Julia Gillard",
        "description": (
            "Prime Minister Julia Gillard of Australia speaks about the U.S. and Australia's "
            "security alliance and strategic outlook on China at a news conference with President Obama."
        ),
    }
]


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def convert_to_wav(ffmpeg_binary: Path, source_path: Path, output_path: Path) -> None:
    command = [
        str(ffmpeg_binary),
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def write_manual_instructions(
    input_dir: Path,
    ffmpeg_binary: Path,
    failures: list[dict[str, str]],
) -> Path:
    lines = [
        "# 공개 사람 음성 샘플 수동 준비 안내",
        "",
        "자동 다운로드가 실패했을 때 아래 절차로 수동 준비할 수 있습니다.",
        "",
    ]
    for item in SAMPLE_SPECS:
        original_name = item["original_filename"]
        target_name = item["output_filename"]
        lines.extend(
            [
                f"## {item['id']}",
                f"- 원본 설명 페이지: {item['source_page']}",
                f"- 직접 다운로드 URL: {item['download_url']}",
                f"- 저장할 입력 파일: `{input_dir / target_name}`",
                f"- 변환 명령:",
                "",
                "```powershell",
                f"{ffmpeg_binary} -y -i \"{original_name}\" -ac 1 -ar 16000 -sample_fmt s16 \"{input_dir / target_name}\"",
                "```",
                "",
            ]
        )
    if failures:
        lines.extend(
            [
                "## 자동 다운로드 실패 기록",
                "",
            ]
        )
        for failure in failures:
            lines.append(f"- {failure['id']}: {failure['reason']}")
    path = input_dir / "MANUAL_DOWNLOAD_INSTRUCTIONS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare public-license human voice samples for pipeline tests.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--ffmpeg-binary", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--keep-originals", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = input_dir / "_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    if not args.ffmpeg_binary.exists():
        failures.append(
            {
                "id": "ffmpeg_missing",
                "reason": f"ffmpeg binary not found: {args.ffmpeg_binary}",
            }
        )

    for sample in SAMPLE_SPECS:
        original_path = downloads_dir / sample["original_filename"]
        output_path = input_dir / sample["output_filename"]
        entry = {
            "id": sample["id"],
            "speaker": sample["speaker"],
            "description": sample["description"],
            "source_page": sample["source_page"],
            "download_url": sample["download_url"],
            "license_name": sample["license_name"],
            "license_url": sample["license_url"],
            "author": sample["author"],
            "original_file": str(original_path),
            "prepared_input_file": str(output_path),
            "status": "pending",
        }

        try:
            if not original_path.exists() or original_path.stat().st_size == 0:
                download_file(sample["download_url"], original_path)
            if not args.ffmpeg_binary.exists():
                raise FileNotFoundError(f"ffmpeg binary not found: {args.ffmpeg_binary}")
            if not output_path.exists() or output_path.stat().st_size == 0:
                convert_to_wav(args.ffmpeg_binary, original_path, output_path)
            entry["status"] = "ready"
            entry["size_bytes"] = output_path.stat().st_size
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = str(exc)
            failures.append({"id": sample["id"], "reason": str(exc)})
        manifest_entries.append(entry)

    manifest = {
        "input_dir": str(input_dir),
        "samples": manifest_entries,
        "failures": failures,
    }
    manifest_path = input_dir / "sample_sources.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_manual_instructions(input_dir, args.ffmpeg_binary, failures)
    if failures:
        failure_path = input_dir / "download_failures.json"
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"manifest": str(manifest_path), "failures": failures}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"manifest": str(manifest_path), "status": "ready"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
