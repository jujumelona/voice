from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Argos Translate language packages")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Language pair as source:target, for example en:ko. Can be repeated.",
    )
    parser.add_argument("--status-json", type=Path)
    args = parser.parse_args()

    pairs = _expand_pairs(args.pair or ["en:ko", "ko:en"])

    import argostranslate.package as package
    import argostranslate.translate as translate

    print("Argos: updating package index", flush=True)
    package.update_package_index()
    print("Argos: reading available packages", flush=True)
    available = package.get_available_packages()
    installed_languages = translate.get_installed_languages()
    installed_pairs = {
        (language.code, translation.to_lang.code)
        for language in installed_languages
        for translation in language.translations_from
    }
    installed = []
    already_installed = []
    missing = []
    download_dir = (
        args.status_json.parent / "downloads" / "argos"
        if args.status_json
        else Path.cwd() / "downloads" / "argos"
    )
    download_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        source, target = _parse_pair(pair)
        if (source, target) in installed_pairs:
            print(f"Argos: already installed {source}:{target}", flush=True)
            already_installed.append(pair)
            continue
        match = next(
            (
                candidate
                for candidate in available
                if candidate.from_code == source and candidate.to_code == target
            ),
            None,
        )
        if match is None:
            print(f"Argos: package unavailable {source}:{target}", flush=True)
            missing.append(pair)
            continue
        path = _download_package(match, download_dir)
        print(f"Argos: installing {source}:{target} from {path}", flush=True)
        package.install_from_path(path)
        installed.append(pair)

    languages = [language.code for language in translate.get_installed_languages()]
    report = {
        "requested": pairs,
        "installed": installed,
        "already_installed": already_installed,
        "missing": missing,
        "languages": languages,
    }
    if args.status_json:
        args.status_json.parent.mkdir(parents=True, exist_ok=True)
        args.status_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(f"Argos packages unavailable: {', '.join(missing)}")


def _parse_pair(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise SystemExit(f"Invalid language pair: {value!r}. Use source:target.")
    return parts[0].strip(), parts[1].strip()


def _expand_pairs(values: list[str]) -> list[str]:
    pairs: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                pairs.append(part)
    return pairs


def _download_package(match, download_dir: Path) -> Path:
    links = [link for link in getattr(match, "links", []) if str(link).startswith(("http://", "https://"))]
    if not links:
        print(f"Argos: downloading {match.from_code}:{match.to_code} with Argos downloader", flush=True)
        return Path(match.download())

    url = links[0]
    name = url.rsplit("/", 1)[-1] or f"{match.code}.argosmodel"
    out = download_dir / name
    if out.exists() and out.stat().st_size > 0:
        print(f"Argos: using cached {out}", flush=True)
        return out
    if out.exists():
        out.unlink()

    print(f"Argos: downloading {match.from_code}:{match.to_code} from {url}", flush=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--output",
                str(out),
                url,
            ],
            check=True,
        )
    else:
        urllib.request.urlretrieve(url, out)
    if not out.exists() or out.stat().st_size <= 0:
        raise SystemExit(f"Argos download produced an empty file: {out}")
    return out


if __name__ == "__main__":
    main()
