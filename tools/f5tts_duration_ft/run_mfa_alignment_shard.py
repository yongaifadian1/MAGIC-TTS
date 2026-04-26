#!/usr/bin/env python3
"""Run MFA on one shard manifest and emit *.words.json sidecars for dataset building."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MFA_BIN = os.environ.get("MAGICTTS_MFA_BIN")
MFA_ROOT = Path(os.environ.get("MAGICTTS_MFA_ROOT", str((REPO_ROOT / ".mfa_root").resolve())))
DEFAULT_AUDIO_ROOT = REPO_ROOT / "data" / "raw_audio"

LANGUAGE_MODELS = {
    "en": {
        "dictionary": MFA_ROOT / "pretrained_models" / "dictionary" / "english_mfa.dict",
        "acoustic": MFA_ROOT / "pretrained_models" / "acoustic" / "english_mfa.zip",
    },
    "zh": {
        "dictionary": MFA_ROOT / "pretrained_models" / "dictionary" / "mandarin_mfa.dict",
        "acoustic": MFA_ROOT / "pretrained_models" / "acoustic" / "mandarin_mfa.zip",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--audio-root", default=str(DEFAULT_AUDIO_ROOT))
    parser.add_argument("--num-jobs", type=int, default=2)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def mfa_env(runtime_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(runtime_root)
    env["PKUSEG_HOME"] = str(MFA_ROOT / "pkuseg")
    return env


def resolve_mfa_bin() -> str:
    if MFA_BIN:
        return MFA_BIN
    path_mfa = shutil.which("mfa")
    if path_mfa:
        return path_mfa
    raise FileNotFoundError(
        "mfa not found in PATH; install montreal-forced-aligner or set MAGICTTS_MFA_BIN"
    )


def load_records(manifest_path: Path) -> list[dict]:
    records = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sidecar_path(sidecar_root: Path, rel_audio_path: str) -> Path:
    return sidecar_root / Path(rel_audio_path).with_suffix(".words.json")


def select_language(records: list[dict]) -> str:
    languages = sorted({record["language"] for record in records})
    if len(languages) != 1:
        raise SystemExit(f"mixed languages in shard manifest: {languages}")
    language = languages[0]
    if language not in LANGUAGE_MODELS:
        raise SystemExit(f"unsupported language: {language}")
    return language


def build_sidecar_payload(record: dict, aligned_json_path: Path) -> dict | None:
    with aligned_json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    entries = obj.get("tiers", {}).get("words", {}).get("entries", [])
    words = []
    for start, end, text in entries:
        text = str(text).strip()
        if not text:
            continue
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        words.append({"word": text, "start": start, "end": end})
    if not words:
        return None
    return {
        "source": "mfa",
        "language": record["language"],
        "audio_path": record["audio_path"],
        "raw_text": record["text"],
        "segments": [
            {
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": record["text"],
                "words": words,
            }
        ],
    }


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    sidecar_root = Path(args.sidecar_root).resolve()
    audio_root = Path(args.audio_root).resolve()
    sidecar_root.mkdir(parents=True, exist_ok=True)
    records = load_records(manifest_path)
    if not records:
        raise SystemExit(f"empty manifest: {manifest_path}")

    language = select_language(records)
    model_info = LANGUAGE_MODELS[language]

    todo_records = []
    missing_audio = 0
    existing = 0
    for record in records:
        target_path = sidecar_path(sidecar_root, record["rel_audio_path"])
        if args.skip_existing and target_path.exists():
            existing += 1
            continue
        if not Path(record["audio_path"]).exists():
            missing_audio += 1
            continue
        todo_records.append(record)

    report_dir = sidecar_root / "_reports" / language
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{manifest_path.stem}.summary.json"

    summary = {
        "manifest": str(manifest_path),
        "language": language,
        "num_jobs": args.num_jobs,
        "requested": len(records),
        "skipped_existing": existing,
        "missing_audio": missing_audio,
        "queued": len(todo_records),
        "succeeded": 0,
        "failed_missing_json": 0,
        "failed_empty_alignment": 0,
        "failed_write": 0,
    }

    if not todo_records:
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    work_base = Path(args.work_dir) if args.work_dir else Path(tempfile.gettempdir())
    work_tag = hashlib.md5(str(manifest_path).encode("utf-8")).hexdigest()[:8]
    shard_work_dir = work_base / f"mfa_{language}_{manifest_path.stem}_{work_tag}"
    if shard_work_dir.exists():
        shutil.rmtree(shard_work_dir)
    corpus_dir = shard_work_dir / "corpus"
    aligned_dir = shard_work_dir / "aligned"
    temp_dir = shard_work_dir / "temp"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    try:
        for record in todo_records:
            utt_id = record["utt_id"]
            audio_path = Path(record["audio_path"]).resolve()
            audio_link = corpus_dir / f"{utt_id}{audio_path.suffix}"
            lab_path = corpus_dir / f"{utt_id}.lab"
            if audio_link.exists() or audio_link.is_symlink():
                audio_link.unlink()
            audio_link.symlink_to(audio_path)
            lab_path.write_text(record["text"].strip() + "\n", encoding="utf-8")

        runtime_root = shard_work_dir / "mfa_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        run(
            [
                resolve_mfa_bin(),
                "align",
                str(corpus_dir),
                str(model_info["dictionary"]),
                str(model_info["acoustic"]),
                str(aligned_dir),
                "--single_speaker",
                "--output_format",
                "json",
                "--clean",
                "-j",
                str(args.num_jobs),
                "-t",
                str(temp_dir),
            ],
            env=mfa_env(runtime_root),
            cwd=shard_work_dir,
        )

        for record in todo_records:
            aligned_json_path = aligned_dir / f"{record['utt_id']}.json"
            if not aligned_json_path.exists():
                summary["failed_missing_json"] += 1
                continue
            payload = build_sidecar_payload(record, aligned_json_path)
            if payload is None:
                summary["failed_empty_alignment"] += 1
                continue

            output_path = sidecar_path(sidecar_root, record["rel_audio_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                summary["succeeded"] += 1
            except OSError:
                summary["failed_write"] += 1
    finally:
        report_dir.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if not args.keep_workdir and shard_work_dir.exists():
            shutil.rmtree(shard_work_dir, ignore_errors=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
