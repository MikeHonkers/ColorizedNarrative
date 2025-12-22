import torchaudio
from torchaudio.functional import resample
import numpy as np
import os

def preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    max_seconds: float | None = 20.0
) -> tuple[np.ndarray, int]:
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path} (cwd={os.getcwd()})")

    waveform, sr = torchaudio.load(audio_path)
    
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if max_seconds is not None and max_seconds > 0:
        max_samples = int(max_seconds * sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

    y = waveform.squeeze(0).numpy()
    y = y.astype(np.float32)

    if sr != target_sr:
        waveform_resampled = resample(waveform, orig_freq=sr, new_freq=target_sr)
        y = waveform_resampled.squeeze(0).numpy().astype(np.float32)
        sr = target_sr

    return y, sr