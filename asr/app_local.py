#!/usr/bin/env python3

from flask import Flask, request, jsonify
import os
import numpy as np
import torch
from pyannote.audio import Pipeline
from transformers import WhisperProcessor
import onnxruntime as ort
from preprocess import preprocess_audio
import requests

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH", "../models/whisper")
CONNECTOR_URL = os.getenv("CONNECTOR_URL", "http://localhost:5001/connector")

encoder_path = os.path.join(MODEL_PATH, "whisper_encoder", "1", "model.onnx")
decoder_path = os.path.join(MODEL_PATH, "whisper_decoder", "1", "model.onnx")

if os.path.exists(encoder_path) and os.path.exists(decoder_path):
    print("[ASR] Loading ONNX models")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    encoder_session = ort.InferenceSession(encoder_path, providers=providers)
    decoder_session = ort.InferenceSession(decoder_path, providers=providers)
    USE_ONNX = True
else:
    print("[ASR] Loading transformers model")
    from transformers import WhisperForConditionalGeneration
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    if torch.cuda.is_available():
        model = model.to("cuda")
    USE_ONNX = False

processor = WhisperProcessor.from_pretrained(
    MODEL_PATH if os.path.exists(os.path.join(MODEL_PATH, "tokenizer_config.json")) else "openai/whisper-small"
)

print("[ASR] Loading diarization")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
if torch.cuda.is_available():
    diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))

print("[ASR] Ready")


def adaptive_max_new_tokens(duration_s: float, scale: float = 4.2, bias: int = 8,
                            hard_min: int = 10, hard_max: int = 160) -> int:
    m = int(scale * float(duration_s) + int(bias))
    return max(int(hard_min), min(int(hard_max), m))


def transcribe_segment_onnx(seg_audio: np.ndarray, sample_rate: int = 16000, max_new_tokens: int = 128) -> str:
    inputs = processor(seg_audio, sampling_rate=sample_rate, return_tensors="np")
    input_features = inputs.input_features.astype(np.float32)
    
    encoder_output = encoder_session.run(None, {"input_features": input_features})
    encoder_hidden_states = encoder_output[0]
    
    decoder_input_ids = np.array([[50258]], dtype=np.int64)
    generated_ids = []
    
    for _ in range(max_new_tokens):
        decoder_output = decoder_session.run(
            None,
            {"input_ids": decoder_input_ids, "encoder_hidden_states": encoder_hidden_states}
        )
        logits = decoder_output[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token)
        
        if next_token == 50257:
            break
        
        decoder_input_ids = np.array([[next_token]], dtype=np.int64)
    
    return processor.batch_decode([generated_ids], skip_special_tokens=True)[0].strip()


def transcribe_segment_torch(seg_audio: np.ndarray, sample_rate: int = 16000, max_new_tokens: int = 128) -> str:
    inputs = processor(seg_audio, sampling_rate=sample_rate, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def transcribe_segment(seg_audio: np.ndarray, sample_rate: int = 16000, max_new_tokens: int = 128) -> str:
    if USE_ONNX:
        return transcribe_segment_onnx(seg_audio, sample_rate, max_new_tokens)
    return transcribe_segment_torch(seg_audio, sample_rate, max_new_tokens)


def build_export_json(diar_segments, audio_filename, sample_rate, duration_sec,
                     asr_model_id="whisper", diar_model_id="pyannote/speaker-diarization-3.1",
                     language: str = "ru"):
    if not diar_segments:
        raise RuntimeError("No speaker segments found")
    
    speaker_map = {}
    next_spk_idx = 0
    segments_json = []
    
    for i, seg in enumerate(diar_segments, start=1):
        raw_spk = str(seg["speaker"])
        if raw_spk not in speaker_map:
            speaker_map[raw_spk] = f"S{next_spk_idx}"
            next_spk_idx += 1
        spk_id = speaker_map[raw_spk]
        
        segments_json.append({
            "id": f"seg_{i:04d}",
            "start": round(float(seg["start"]), 2),
            "end": round(float(seg["end"]), 2),
            "speaker": spk_id,
            "speaker_raw": raw_spk,
            "text": str(seg.get("text", "")).strip(),
            "conf": None,
            "n_best": [],
            "uncertain_spans": [],
        })
    
    return {
        "schema_version": 1,
        "language": language,
        "meta": {
            "source": audio_filename,
            "sample_rate": int(sample_rate),
            "duration": duration_sec,
            "num_speakers": len(speaker_map),
            "speaker_map": speaker_map,
            "asr_model": asr_model_id,
            "diarization_model": diar_model_id,
        },
        "segments": segments_json,
    }


@app.route('/asr', methods=['POST'])
def asr():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    uploaded_file = request.files['audio']
    original_filename = uploaded_file.filename or "audio.wav"
    temp_path = "/tmp/input_audio.wav"
    uploaded_file.save(temp_path)
    
    try:
        print(f"[ASR] Processing {original_filename}")
        audio, sr = preprocess_audio(temp_path)
        duration_sec = len(audio) / sr
        print(f"[ASR] Duration: {duration_sec:.2f}s")
        
        print("[ASR] Diarization")
        waveform = torch.from_numpy(audio).unsqueeze(0)
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sr})
        
        transcription = []
        
        for turn, speaker in diarization.exclusive_speaker_diarization:
            start, end = turn.start, turn.end
            if end - start < 0.3:
                continue
            
            start_idx, end_idx = int(start * sr), int(end * sr)
            if end_idx - start_idx < int(0.3 * sr):
                continue
            
            seg_audio = audio[start_idx:end_idx]
            duration_seg = (end_idx - start_idx) / sr
            max_new = adaptive_max_new_tokens(duration_seg)
            
            print(f"[ASR] [{start:.2f}s-{end:.2f}s] {speaker}")
            text = transcribe_segment(seg_audio, sample_rate=sr, max_new_tokens=max_new)
            
            transcription.append({
                "speaker": str(speaker),
                "start": start,
                "end": end,
                "text": text,
                "duration": float(duration_seg),
            })
        
        transcription.sort(key=lambda s: s["start"])
        print(f"[ASR] Transcribed {len(transcription)} segments")
        
        result_json = build_export_json(
            diar_segments=transcription,
            audio_filename=original_filename,
            sample_rate=sr,
            duration_sec=duration_sec,
        )
        
        print("[ASR] Sending to connector")
        try:
            response = requests.post(CONNECTOR_URL, json=result_json, timeout=180)
            response.raise_for_status()
            return jsonify(response.json()), 200
        except requests.exceptions.RequestException as e:
            print(f"[ASR] Connector error: {e}")
            return jsonify({"error": "Connector failed", "diarized_result": result_json}), 502
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "asr"}), 200


if __name__ == '__main__':
    print("[ASR] http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
