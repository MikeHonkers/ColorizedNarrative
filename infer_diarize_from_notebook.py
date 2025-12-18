#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script: Whisper (base + LoRA adapter via PEFT) + Pyannote diarization,
then export diarized transcript to JSON (schema_version=1).

Example:
  python infer_diarize_export.py --audio "videoplayback.m4a" --base_dir "W:\whisper_sova" --out_json "videoplayback_20s_diarized.json"

Notes:
- Requires: torch, transformers, peft, pyannote.audio, librosa, soundfile, numpy
- For Pyannote diarization you need a Hugging Face token with access to the model:
    set HF_TOKEN=hf_...
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import librosa

import torch
from transformers import pipeline as hf_pipeline, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from pyannote.audio import Pipeline as PyannotePipeline


def resample_to(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample mono float32 audio to target_sr."""
    if sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)


def save_tmp_wav(y: np.ndarray, sr: int, out_dir: str, fname: str) -> str:
    """Write temporary wav and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    sf.write(path, y, sr)
    return path


def adaptive_max_new_tokens(duration_s: float, scale: float = 4.2, bias: int = 8,
                            hard_min: int = 10, hard_max: int = 160) -> int:
    """Heuristic for max_new_tokens as a function of segment duration."""
    m = int(scale * float(duration_s) + int(bias))
    return max(int(hard_min), min(int(hard_max), m))


def load_audio_file(audio_path: str, max_seconds: Optional[float] = None) -> tuple[np.ndarray, int]:
    """Load audio via librosa. Returns mono float32 and sr."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path} (cwd={os.getcwd()})")

    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim > 1:
        y = y.mean(axis=0)  # (channels, samples) -> mono
    y = y.astype(np.float32)

    if max_seconds is not None and max_seconds > 0:
        max_samples = int(max_seconds * sr)
        if y.shape[0] > max_samples:
            y = y[:max_samples]
    return y, int(sr)


@dataclass
class Models:
    asr_pipe: Any
    diar_pipe: Any


def build_asr_pipe(
    processor_dir: str,
    model_id: str,
    model_cache_dir: str,
    adapter_dir: str,
    language: str = "russian",
) -> Any:
    """Build HuggingFace ASR pipeline using Whisper + LoRA adapter (PEFT)."""
    processor = WhisperProcessor.from_pretrained(processor_dir)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=model_cache_dir,
        device_map="auto",
    )
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    ft_model.eval()

    pipe = hf_pipeline(
        task="automatic-speech-recognition",
        model=ft_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        generate_kwargs={
            "task": "transcribe",
            "language": language,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.0,
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 1.30,
            "length_penalty": 0.0,
            "return_timestamps": False,
        },
    )
    return pipe


def build_diar_pipe(hf_token: Optional[str]) -> Any:
    """Build Pyannote diarization pipeline."""
    diar = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diar.to(device)
    print("Diarization initialized on:", device)
    return diar


def diarize_and_transcribe(
    y: np.ndarray,
    sr: int,
    global_wav_path: str,
    diar_pipeline: Any,
    asr_pipe: Any,
    min_seg_dur: float = 0.3,
) -> List[Dict[str, Any]]:
    """Run diarization on global_wav_path, then ASR for each speaker segment."""
    diar_out = diar_pipeline(global_wav_path)
    ann = diar_out.speaker_diarization if hasattr(diar_out, "speaker_diarization") else diar_out

    segments: List[Dict[str, Any]] = []
    for turn, _, speaker in ann.itertracks(yield_label=True):
        start_s = float(turn.start)
        end_s = float(turn.end)
        if end_s <= start_s:
            continue
        if (end_s - start_s) < float(min_seg_dur):
            continue

        start_idx = int(start_s * sr)
        end_idx = int(end_s * sr)
        if end_idx <= start_idx:
            continue

        seg = y[start_idx:end_idx]
        if seg.size < int(min_seg_dur * sr):
            continue

        duration_seg = (end_idx - start_idx) / sr
        max_new = adaptive_max_new_tokens(duration_seg)

        asr_out = asr_pipe(
            {"array": seg, "sampling_rate": sr},
            generate_kwargs=dict(max_new_tokens=max_new),
        )
        text = (asr_out.get("text") or "").strip()

        segments.append(
            {
                "speaker": str(speaker),
                "start": start_s,
                "end": end_s,
                "text": text,
                "duration": float(duration_seg),
                "max_new_tokens": int(max_new),
            }
        )

    segments.sort(key=lambda s: s["start"])
    return segments


def export_segments_json(
    diar_segments: List[Dict[str, Any]],
    out_json_path: str,
    audio_file: str,
    sample_rate: int,
    duration_sec: Optional[float],
    asr_model_id: str,
    diar_model_id: str = "pyannote/speaker-diarization-3.1",
    language: str = "ru",
) -> Dict[str, Any]:
    """Export diarized segments to JSON structure matching your notebook."""
    if not diar_segments:
        raise RuntimeError("diar_segments is empty — no speaker segments found (maybe only noise?).")

    speaker_map: Dict[str, str] = {}
    next_spk_idx = 0
    segments_json: List[Dict[str, Any]] = []

    for i, seg in enumerate(diar_segments, start=1):
        raw_spk = str(seg["speaker"])
        if raw_spk not in speaker_map:
            speaker_map[raw_spk] = f"S{next_spk_idx}"
            next_spk_idx += 1
        spk_id = speaker_map[raw_spk]

        seg_id = f"seg_{i:04d}"
        start_t = round(float(seg["start"]), 2)
        end_t = round(float(seg["end"]), 2)

        segments_json.append(
            {
                "id": seg_id,
                "start": start_t,
                "end": end_t,
                "speaker": spk_id,
                "speaker_raw": raw_spk,
                "text": str(seg.get("text", "")),
                "conf": None,
                "n_best": [],
                "uncertain_spans": [],
            }
        )

    export_obj: Dict[str, Any] = {
        "schema_version": 1,
        "language": language,
        "meta": {
            "source": audio_file,
            "sample_rate": int(sample_rate),
            "duration": duration_sec,
            "num_speakers": len(speaker_map),
            "speaker_map": speaker_map,
            "asr_model": asr_model_id,
            "diarization_model": diar_model_id,
        },
        "segments": segments_json,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json_path)) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(export_obj, f, ensure_ascii=False, indent=2)

    return export_obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper+LoRA inference + diarization + JSON export")
    p.add_argument("--audio", required=True, help="Path to audio file (e.g., .m4a/.wav/.mp3)")
    p.add_argument("--base_dir", default=r"W:\whisper_sova", help="Base directory")
    p.add_argument("--model_id", default="openai/whisper-small", help="Base Whisper model id")
    p.add_argument("--target_sr", type=int, default=16000, help="Target sample rate")
    p.add_argument("--language", default="russian", help='Whisper language setting (e.g., "russian")')
    p.add_argument("--max_seconds", type=float, default=20.0, help="Trim: max seconds from start (0=disable)")
    p.add_argument("--min_seg_dur", type=float, default=0.3, help="Min diarization segment duration (seconds)")

    p.add_argument("--output_dir", default=None, help="Output dir (default: <base_dir>\\output)")
    p.add_argument("--processor_dir", default=None, help="Processor dir (default: <output_dir>\\processor)")
    p.add_argument("--adapter_dir", default=None, help="LoRA adapter dir (default: <output_dir>\\lora_adapter_fast)")
    p.add_argument("--model_cache_dir", default=None, help="Model cache dir (default: <base_dir>\\model)")

    p.add_argument("--tmp_wav_name", default="_tmp_for_diar.wav", help="Filename for temp wav in output_dir")
    p.add_argument("--out_json", default=None, help="Output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir or os.path.join(base_dir, "output")
    processor_dir = args.processor_dir or os.path.join(output_dir, "processor")
    adapter_dir = args.adapter_dir or os.path.join(output_dir, "lora_adapter_fast")
    model_cache_dir = args.model_cache_dir or os.path.join(base_dir, "model")

    max_seconds = None if (args.max_seconds is None or args.max_seconds <= 0) else float(args.max_seconds)

    y, sr = load_audio_file(args.audio, max_seconds=max_seconds)
    dur_raw = len(y) / sr if sr > 0 else None

    y16 = resample_to(y, sr, int(args.target_sr))
    dur16 = len(y16) / int(args.target_sr)

    wav_path = save_tmp_wav(y16, int(args.target_sr), out_dir=output_dir, fname=args.tmp_wav_name)

    print(f"Audio: {args.audio}")
    print(f"  raw sr={sr}, dur≈{dur_raw:.2f}s" + (f" (trimmed to {max_seconds:.2f}s)" if max_seconds else ""))
    print(f"  resampled: sr={args.target_sr}, dur≈{dur16:.2f}s")
    print(f"  temp wav: {wav_path}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN env var is not set. Pyannote may fail if model access requires auth.")

    asr_pipe = build_asr_pipe(
        processor_dir=processor_dir,
        model_id=args.model_id,
        model_cache_dir=model_cache_dir,
        adapter_dir=adapter_dir,
        language=args.language,
    )
    diar_pipe = build_diar_pipe(hf_token=hf_token)

    max_new_full = adaptive_max_new_tokens(dur16, hard_max=160)
    asr_full = asr_pipe(wav_path, generate_kwargs=dict(max_new_tokens=max_new_full))
    full_text = (asr_full.get("text") or "").strip()
    print("\n=== Full transcription (no diarization) ===")
    print(full_text)

    diar_segments = diarize_and_transcribe(
        y16,
        int(args.target_sr),
        wav_path,
        diar_pipeline=diar_pipe,
        asr_pipe=asr_pipe,
        min_seg_dur=float(args.min_seg_dur),
    )

    print(f"\n=== Diarization + ASR segments (min_seg_dur={args.min_seg_dur}) ===")
    if not diar_segments:
        print("<no speech segments / only noise>")
    else:
        for seg in diar_segments:
            print(
                f'[spk {seg["speaker"]}] '
                f'{seg["start"]:6.2f}-{seg["end"]:6.2f}s '
                f'(dur={seg["duration"]:.2f}s, max_new={seg["max_new_tokens"]})'
            )
            print("   ", seg["text"])

    if args.out_json:
        out_json_path = args.out_json
    else:
        base = os.path.splitext(os.path.basename(args.audio))[0]
        suffix = f"_{int(max_seconds)}s" if (max_seconds and max_seconds > 0) else ""
        out_json_path = os.path.join(output_dir, f"{base}{suffix}_diarized.json")

    export_obj = export_segments_json(
        diar_segments=diar_segments,
        out_json_path=out_json_path,
        audio_file=args.audio,
        sample_rate=int(args.target_sr),
        duration_sec=float(dur16),
        asr_model_id=args.model_id,
        diar_model_id="pyannote/speaker-diarization-3.1",
        language="ru",
    )

    print(f"\nExported diarized transcript to: {out_json_path}")
    print("\nExample (first 3 segments):")
    preview = {**export_obj, "segments": export_obj["segments"][:3]}
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
