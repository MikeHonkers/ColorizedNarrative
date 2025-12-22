from flask import Flask, request, jsonify
import os
import numpy as np
import torch
from pyannote.audio import Pipeline
import tritonclient.http as httpclient
from transformers import pipeline, WhisperProcessor
from preprocess import preprocess_audio
import requests

app = Flask(__name__)

TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")
HF_TOKEN = os.getenv("HF_TOKEN")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

processor = WhisperProcessor.from_pretrained("/models/whisper")

WHISPER_ENCODER_MODEL = "whisper_encoder"
WHISPER_DECODER_MODEL = "whisper_decoder"

CONNECTOR_URL = "http://connector:5001/connector"

def load_whisper_models():
    for model_name in [WHISPER_ENCODER_MODEL, WHISPER_DECODER_MODEL]:
        if not triton_client.is_model_ready(model_name):
            print(f"Loading model: {model_name}")
            triton_client.load_model(model_name)

def unload_whisper_models():
    for model_name in [WHISPER_ENCODER_MODEL, WHISPER_DECODER_MODEL]:
        if triton_client.is_model_ready(model_name):
            print(f"Unloading model: {model_name}")
            triton_client.unload_model(model_name)

def adaptive_max_new_tokens(duration_s: float, scale: float = 4.2, bias: int = 8,
                            hard_min: int = 10, hard_max: int = 160) -> int:
    """Heuristic for max_new_tokens as a function of segment duration."""
    m = int(scale * float(duration_s) + int(bias))
    return max(int(hard_min), min(int(hard_max), m))

def transcribe_segment(seg_audio: np.ndarray, sample_rate: int = 16000, max_new_tokens: int = 128) -> str:
    inputs = processor(seg_audio, sampling_rate=sample_rate, return_tensors="np")
    input_features = inputs.input_features.astype(np.float32)

    encoder_input = httpclient.InferInput("input_features", input_features.shape, "FP32")
    encoder_input.set_data_from_numpy(input_features)
    encoder_result = triton_client.infer(
        model_name=WHISPER_ENCODER_MODEL,
        inputs=[encoder_input]
    )
    encoder_hidden_states = encoder_result.as_numpy("last_hidden_states")

    decoder_input_ids = np.array([[50258]], dtype=np.int64)
    generated_ids = []

    for _ in range(max_new_tokens):
        dec_inputs = [
            httpclient.InferInput("input_ids", decoder_input_ids.shape, "INT64"),
            httpclient.InferInput("last_hidden_state", encoder_hidden_states.shape, "FP32")
        ]
        dec_inputs[0].set_data_from_numpy(decoder_input_ids)
        dec_inputs[1].set_data_from_numpy(encoder_hidden_states)

        dec_result = triton_client.infer(
            model_name=WHISPER_DECODER_MODEL,
            inputs=dec_inputs
        )
        logits = dec_result.as_numpy("logits")

        next_token = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token)

        if next_token in [50257]:
            break

        decoder_input_ids = np.array([[next_token]], dtype=np.int64)

    text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
    return text.strip()

def build_export_json(
    diar_segments,
    audio_filename,
    sample_rate,
    duration_sec,
    asr_model_id = "custom-onnx-whisper-small",
    diar_model_id = "pyannote/speaker-diarization-3.1",
    language: str = "ru",
):
    if not diar_segments:
        raise RuntimeError("No speaker segments found after diarization and transcription.")

    speaker_map = {}
    next_spk_idx = 0
    segments_json = []

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
                "text": str(seg.get("text", "")).strip(),
                "conf": None,
                "n_best": [],
                "uncertain_spans": [],
            }
        )

    export_obj = {
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

    return export_obj

@app.route('/asr', methods=['POST'])
def asr():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    uploaded_file = request.files['audio']
    original_filename = uploaded_file.filename or "uploaded_audio.wav"
    temp_path = "/tmp/input_audio.wav"
    uploaded_file.save(temp_path)

    try:
        load_whisper_models()

        audio, sr = preprocess_audio(temp_path)
        duration_sec = len(audio) / sr if len(audio) > 0 else 0.0

        waveform = torch.from_numpy(audio).unsqueeze(0)
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sr})

        transcription = []

        for turn, speaker in diarization.exclusive_speaker_diarization:
            start = turn.start
            end = turn.end

            if end - start < 0.3:
                continue

            start_idx = int(start * sr)
            end_idx = int(end * sr)
            if end_idx <= start_idx or end_idx - start_idx < int(0.3 * sr):
                continue

            seg_audio = audio[start_idx:end_idx]
            duration_seg = (end_idx - start_idx) / sr
            max_new = adaptive_max_new_tokens(duration_seg)

            text = transcribe_segment(seg_audio, sample_rate=sr, max_new_tokens=max_new)

            transcription.append({
                "speaker": str(speaker),
                "start": start,
                "end": end,
                "text": text,
                "duration": float(duration_seg),
                "max_new_tokens": int(max_new),
            })

        transcription.sort(key=lambda s: s["start"])

        result_json = build_export_json(
            diar_segments=transcription,
            audio_filename=original_filename,
            sample_rate=sr,
            duration_sec=duration_sec,
        )

        unload_whisper_models()

        try:
            response = requests.post(
                CONNECTOR_URL,
                json=result_json,
                timeout=120
            )
            response.raise_for_status()
            final_result = response.json()

        except requests.exceptions.RequestException as e:
            print(f"Connector error: {e}")
            return jsonify({
                "error": "Scene generation failed",
                "diarized_result": result_json,
                "connector_error": str(e)
            }), 502

        return jsonify(final_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)