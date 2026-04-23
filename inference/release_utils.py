#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "vendor" / "f5tts_duration_ft"


DEFAULT_SYNTH_PYTHON = os.environ.get("MAGICTTS_SYNTH_PYTHON", sys.executable)
SYNTH_SCRIPT = TOOL_ROOT / "ttrack_edit_synthesize.py"


def build_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    prefix = env.get("CONDA_PREFIX") or sys.prefix
    lib_dir = Path(prefix).resolve() / "lib"
    if lib_dir.exists():
        lib_str = str(lib_dir)
        current = env.get("LD_LIBRARY_PATH", "")
        if current:
            parts = current.split(":")
            if lib_str not in parts:
                env["LD_LIBRARY_PATH"] = f"{lib_str}:{current}"
        else:
            env["LD_LIBRARY_PATH"] = lib_str
    return env


def _load_internal_api():
    if str(TOOL_ROOT) not in sys.path:
        sys.path.insert(0, str(TOOL_ROOT))

    from custom_prefix_showcase_demos import build_custom_track  # type: ignore
    from custom_prefix_showcase_demos import frames_from_ms  # type: ignore
    from run_timing_control_accuracy_b150 import HOP_LENGTH  # type: ignore
    from run_timing_control_accuracy_b150 import TARGET_SAMPLE_RATE  # type: ignore
    from run_timing_control_accuracy_b150 import add_residual_to_last_pause  # type: ignore
    from run_timing_control_accuracy_b150 import build_token_track_from_words  # type: ignore
    from run_timing_control_accuracy_b150 import load_words_from_mfa_output  # type: ignore
    from run_timing_control_accuracy_b150 import run_mfa_for_language  # type: ignore
    from run_timing_control_accuracy_b150 import write_json  # type: ignore

    return {
        "build_custom_track": build_custom_track,
        "frames_from_ms": frames_from_ms,
        "HOP_LENGTH": HOP_LENGTH,
        "TARGET_SAMPLE_RATE": TARGET_SAMPLE_RATE,
        "add_residual_to_last_pause": add_residual_to_last_pause,
        "build_token_track_from_words": build_token_track_from_words,
        "load_words_from_mfa_output": load_words_from_mfa_output,
        "run_mfa_for_language": run_mfa_for_language,
        "write_json": write_json,
    }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    api = _load_internal_api()
    api["write_json"](path, payload)


def load_preset(path: Path) -> Dict[str, Any]:
    return load_json(path)


def select_variants(preset: Dict[str, Any], variant_slugs: Optional[str]) -> List[Dict[str, Any]]:
    variants = list(preset.get("variants", []))
    if not variant_slugs:
        return variants
    wanted = {item.strip() for item in variant_slugs.split(",") if item.strip()}
    return [variant for variant in variants if variant["slug"] in wanted]


def prompt_num_frames(audio_path: Path) -> float:
    api = _load_internal_api()
    import torchaudio

    audio, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != api["TARGET_SAMPLE_RATE"]:
        audio = torchaudio.functional.resample(audio, sample_rate, api["TARGET_SAMPLE_RATE"])
    return float(audio.shape[-1]) / api["HOP_LENGTH"]


def load_prompt_audio_24k_mono(audio_path: Path):
    api = _load_internal_api()
    import torch
    import torchaudio

    audio, sample_rate = torchaudio.load(str(audio_path))
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sample_rate != api["TARGET_SAMPLE_RATE"]:
        audio = torchaudio.functional.resample(audio, sample_rate, api["TARGET_SAMPLE_RATE"])
    return audio, api["TARGET_SAMPLE_RATE"]


def build_prompt_prefix_context(
    prompt_audio: Path,
    prompt_text: str,
    prompt_language: str,
    output_dir: Path,
    prompt_id: Optional[str],
    mfa_jobs: int,
    keep_mfa_workdir: bool,
) -> Dict[str, Any]:
    api = _load_internal_api()
    temp_dir = tempfile.mkdtemp(prefix="magictts_release_prompt_mfa_")
    try:
        alignment_root = Path(temp_dir).resolve() / "prompt_alignment_runtime"
        sample_dir = alignment_root / "prompt_alignment"
        sample_dir.mkdir(parents=True, exist_ok=True)

        prompt_wav_path = sample_dir / "prompt_target_only.wav"
        audio_24k, sample_rate = load_prompt_audio_24k_mono(prompt_audio)
        import torchaudio

        torchaudio.save(str(prompt_wav_path), audio_24k, sample_rate)

        spec = SimpleNamespace(
            sample_id="prompt_alignment",
            language=prompt_language,
            sample_dir=sample_dir,
            target_text=prompt_text,
        )
        args = SimpleNamespace(output_prefix="prompt", result_tag="")
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
        api["run_mfa_for_language"](
            samples=[spec],
            output_dir=alignment_root,
            language=prompt_language,
            num_jobs=mfa_jobs,
            keep_workdir=keep_mfa_workdir,
            args=args,
        )

        aligned_raw = load_json(sample_dir / "prompt_target_only_mfa_raw.json")
        words = api["load_words_from_mfa_output"](aligned_raw)
        prefix_tokens, prefix_durations = api["build_token_track_from_words"](words)

        actual_prompt_frames = prompt_num_frames(prompt_audio)
        residual_frames = actual_prompt_frames - float(sum(sum(pair) for pair in prefix_durations))
        api["add_residual_to_last_pause"](prefix_durations, residual_frames)

        prompt_track = {
            "prompt_audio": str(prompt_audio.resolve()),
            "prompt_text": prompt_text,
            "language": prompt_language,
            "prompt_id": prompt_id or prompt_audio.stem,
            "prefix_tokens": prefix_tokens,
            "prefix_durations": prefix_durations,
            "prompt_frames": actual_prompt_frames,
            "num_words": len(words),
            "residual_frames_added_to_last_pause": residual_frames,
        }
        save_json(output_dir / "prompt_track.json", prompt_track)
        save_json(output_dir / "prompt_alignment_raw.json", aligned_raw)
        save_json(
            output_dir / "prompt_alignment_debug.json",
            {
                "alignment_root": str(alignment_root),
                "sample_dir": str(sample_dir),
                "kept_workdir": bool(keep_mfa_workdir),
            },
        )

        return {
            "audio_path": str(prompt_audio.resolve()),
            "prefix_tokens": list(prefix_tokens),
            "prefix_durations": [list(pair) for pair in prefix_durations],
            "prefix_token_count": len(prefix_tokens),
            "source_sample_dir": str(prompt_audio.resolve().parent),
            "source_prompt_frames": actual_prompt_frames,
            "source_total_frames": actual_prompt_frames,
            "prompt_text_source": prompt_text,
            "prompt_id": prompt_id or prompt_audio.stem,
            "prompt_metadata": None,
        }
    finally:
        if not keep_mfa_workdir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def synthesize_with_track(
    prompt_audio: Optional[Path],
    track_json: Path,
    output_dir: Path,
    checkpoint: Path,
    synth_python: str,
    steps: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    seed: Optional[int],
) -> None:
    track = load_json(track_json)
    if prompt_audio is None:
        track_audio = track.get("audio_path")
        if not track_audio:
            raise ValueError("track_json does not contain audio_path; please provide --audio explicitly")
        prompt_audio = Path(track_audio).resolve()
    prompt_frames = float(track["source_prompt_frames"])
    cmd = [
        synth_python,
        str(SYNTH_SCRIPT),
        "--audio",
        str(prompt_audio.resolve()),
        "--track-json",
        str(track_json.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--prompt-frames",
        str(prompt_frames),
        "--steps",
        str(steps),
        "--cfg-strength",
        str(cfg_strength),
        "--sway-sampling-coef",
        str(sway_sampling_coef),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--preserve-cond-audio-in-output",
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=build_subprocess_env())


def write_release_summary(
    output_dir: Path,
    preset: Dict[str, Any],
    prompt_audio: Path,
    prompt_text: str,
    checkpoint: Path,
    variant_dirs: Iterable[Path],
) -> None:
    payload = {
        "preset_slug": preset["slug"],
        "preset_title": preset.get("title"),
        "prompt_audio": str(prompt_audio.resolve()),
        "prompt_text": prompt_text,
        "checkpoint": str(checkpoint.resolve()),
        "variant_dirs": [str(path.resolve()) for path in variant_dirs],
    }
    save_json(output_dir / "summary.json", payload)


def build_track_from_preset(
    prefix_context: Dict[str, Any],
    variant: Dict[str, Any],
    content_ms: float,
    punct_ms: float,
) -> Dict[str, Any]:
    api = _load_internal_api()
    return api["build_custom_track"](
        prefix_context=prefix_context,
        variant=variant,
        content_frames=api["frames_from_ms"](content_ms),
        punct_frames=api["frames_from_ms"](punct_ms),
    )
