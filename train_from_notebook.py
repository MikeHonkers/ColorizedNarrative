#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import annotations

"""
Whisper-small/base + PEFT/LoRA на SOVA-audiobooks-100k
- Без torchcodec: Audio(decode=False) + свой коллатор (soundfile/librosa)
- 30s crop, приводим к (80, 3000) — как требует Whisper
- Патч PEFT forward: не передавать input_ids в Whisper (только input_features)
"""

import os, re, io, json, random
from typing import List, Dict


import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default=r"W:\whisper_sova", help="Base working directory (was BASE_DIR in notebook)")
    return p.parse_args()

args = parse_args()

BASE_DIR   = args.base_dir
MODEL_DIR  = fr"{BASE_DIR}\model"
DATA_DIR   = fr"{BASE_DIR}\data"
OUTPUT_DIR = fr"{BASE_DIR}\output"
CACHE_DIR  = fr"{BASE_DIR}\hf_cache"

for p in [BASE_DIR, MODEL_DIR, DATA_DIR, OUTPUT_DIR, CACHE_DIR]:
    os.makedirs(p, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = fr"{CACHE_DIR}\datasets"
os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR
os.environ["DATASETS_DISABLE_MP"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import soundfile as sf
import librosa
import evaluate
import requests

from datasets import load_dataset, Audio, config as ds_config
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.training_args import TrainingArguments
from transformers.models.whisper.configuration_whisper import WhisperConfig

from peft import LoraConfig, get_peft_model
import peft.peft_model as peft_model_mod

ds_config.TORCHCODEC_AVAILABLE = False

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
print("bf16 supported:", use_bf16)

print("→ Загружаем SOVA-audiobooks-100k ...")
DATASET_ID = "MikeHonkers/SOVA-audiobooks-100k"
ds = load_dataset(DATASET_ID, cache_dir=DATA_DIR, split="train")

ds = ds.train_test_split(test_size=0.02, seed=42)
train_ds, test_holdout = ds["train"], ds["test"]
val_test = test_holdout.train_test_split(test_size=0.5, seed=42)
val_ds, test_ds = val_test["train"], val_test["test"]

train_ds = train_ds.cast_column("audio", Audio(decode=False))
val_ds   = val_ds.cast_column("audio",   Audio(decode=False))
test_ds  = test_ds.cast_column("audio",  Audio(decode=False))

def normalize_ru_text(s: str) -> str:
    s = s.strip().replace("ё", "е")
    s = re.sub(r'[“”«»]', '"', s)
    s = re.sub(r"\s+", " ", s)
    return s

def prepare_batched(batch):
    return {"sentence": [normalize_ru_text(t) for t in batch["text"]]}

print("→ Нормализуем текст ...")
train_ds = train_ds.map(prepare_batched, batched=True, desc="prepare train")
val_ds   = val_ds.map(prepare_batched,   batched=True, desc="prepare val")
test_ds  = test_ds.map(prepare_batched,  batched=True, desc="prepare test")

def has_audio_ref(ex):
    a = ex.get("audio", {})
    return (a.get("path") or a.get("bytes")) is not None

train_ds = train_ds.filter(has_audio_ref)
val_ds   = val_ds.filter(has_audio_ref)
test_ds  = test_ds.filter(has_audio_ref)

TARGET_SR = 16000
MIN_BYTES = int(0.20 * TARGET_SR * 2)

def looks_nonempty(ex):
    a = ex.get("audio", {})
    if a.get("bytes"):
        return len(a["bytes"]) >= MIN_BYTES
    p = a.get("path")
    return bool(p) and os.path.exists(p) and os.path.getsize(p) >= MIN_BYTES

train_ds = train_ds.filter(looks_nonempty)
val_ds   = val_ds.filter(looks_nonempty)
test_ds  = test_ds.filter(looks_nonempty)

print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

MODEL_ID = "openai/whisper-small"

print("→ Загружаем процессор/модель ...")
processor = WhisperProcessor.from_pretrained(
    MODEL_ID, language="russian", task="transcribe", cache_dir=MODEL_DIR
)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, cache_dir=MODEL_DIR, device_map="auto"
)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="russian", task="transcribe"
)

base_cfg = WhisperConfig.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR)
model.config.suppress_tokens = base_cfg.suppress_tokens
model.config.begin_suppress_tokens = base_cfg.begin_suppress_tokens

model.config.use_cache = False

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
if torch.cuda.is_available():
    model.to("cuda")

def _whisper_safe_forward(
    self,
    input_ids=None,
    attention_mask=None,
    inputs_embeds=None,
    decoder_input_ids=None,
    decoder_attention_mask=None,
    decoder_inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    task_ids=None,
    **kwargs,
):
    base_kwargs = {}
    if "input_features" in kwargs and kwargs["input_features"] is not None:
        base_kwargs["input_features"] = kwargs["input_features"]
    if attention_mask is not None:
        base_kwargs["attention_mask"] = attention_mask
    if decoder_input_ids is not None:
        base_kwargs["decoder_input_ids"] = decoder_input_ids
    if decoder_attention_mask is not None:
        base_kwargs["decoder_attention_mask"] = decoder_attention_mask
    if decoder_inputs_embeds is not None:
        base_kwargs["decoder_inputs_embeds"] = decoder_inputs_embeds
    if labels is not None:
        base_kwargs["labels"] = labels
    if output_attentions is not None:
        base_kwargs["output_attentions"] = output_attentions
    if output_hidden_states is not None:
        base_kwargs["output_hidden_states"] = output_hidden_states
    if return_dict is not None:
        base_kwargs["return_dict"] = return_dict
    for k in (
        "use_cache",
        "head_mask",
        "decoder_head_mask",
        "cross_attn_head_mask",
        "encoder_outputs",
        "decoder_position_ids",
        "past_key_values",
        "cache_position",
    ):
        if k in kwargs and kwargs[k] is not None:
            base_kwargs[k] = kwargs[k]

    return self.model(**base_kwargs)

peft_model_mod.PeftModelForSeq2SeqLM.forward = _whisper_safe_forward

random.seed(42)
np.random.seed(42)

wer_metric = evaluate.load("wer")

MAX_FRAMES = 3000
WIN_SAMPLES = int(30.0 * TARGET_SR)

def crop_30s(y: np.ndarray) -> np.ndarray:
    if len(y) > WIN_SAMPLES:
        start = np.random.randint(0, len(y) - WIN_SAMPLES + 1)
        return y[start:start+WIN_SAMPLES]
    return y

def _pad_trunc_features(feats: torch.Tensor) -> torch.Tensor:
    B, C, T = feats.shape
    if T == MAX_FRAMES:
        return feats
    if T > MAX_FRAMES:
        return feats[:, :, :MAX_FRAMES]
    pad = torch.zeros((B, C, MAX_FRAMES - T), dtype=feats.dtype, device=feats.device)
    return torch.cat([feats, pad], dim=-1)

def load_wav_entry(audio_field: dict) -> np.ndarray:
    if audio_field.get("bytes"):
        y, sr = sf.read(io.BytesIO(audio_field["bytes"]), always_2d=False)
    else:
        p = audio_field.get("path")
        if not p:
            raise FileNotFoundError("no audio bytes/path")
        if isinstance(p, str) and p.startswith(("http://", "https://")):
            r = requests.get(p, timeout=60)
            r.raise_for_status()
            y, sr = sf.read(io.BytesIO(r.content), always_2d=False)
        else:
            y, sr = sf.read(p, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    if y.size == 0:
        y = np.zeros(int(0.1 * TARGET_SR), dtype=np.float32)
    return y

def data_collator(batch):
    audios = [crop_30s(load_wav_entry(b["audio"])) for b in batch]

    feats = processor.feature_extractor(
        audios,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        return_attention_mask=False,
    )
    input_features = feats.input_features

    input_features = _pad_trunc_features(input_features)

    labs = processor.tokenizer(
        [b["sentence"] for b in batch],
        return_tensors="pt",
        padding=True,
    )
    labels = labs.input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {"input_features": input_features, "labels": labels}

def compute_metrics(pred):
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_ids = np.argmax(pred_ids, axis=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(references=label_str, predictions=pred_str)}

batch0 = data_collator([train_ds[i] for i in range(min(4, len(train_ds)))])
print("→ Collator OK:", tuple(batch0["input_features"].shape))  # (B, 80, 3000)
for k, v in batch0.items():
    if isinstance(v, torch.Tensor):
        batch0[k] = v.to(model.device)
with torch.amp.autocast("cuda", enabled=torch.cuda.is_available() and not use_bf16):
    out = model(**batch0)
print("Sanity loss:", float(out.loss.detach()))

class NoEvalCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kw): return control

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=300,
    max_steps=8000,
    logging_steps=200,
    fp16=not use_bf16,
    bf16=use_bf16,
    gradient_checkpointing=False,
    dataloader_num_workers=0,
    report_to="none",
    remove_unused_columns=False,
    save_strategy="steps",
    save_steps=4000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[NoEvalCallback()],
)
trainer.args.label_names = ["labels"]

def _noreset_save(self, *a, **kw):
    print("Skip checkpoint body to avoid dataloader reset")
trainer._save_checkpoint = _noreset_save.__get__(trainer, Trainer)

print("→ Trainable %:",
      100 * sum(p.numel() for p in model.parameters() if p.requires_grad) /
      sum(p.numel() for p in model.parameters()))
print("→ Начинаем обучение LoRA ...")
trainer.train()

adapter_dir   = fr"{OUTPUT_DIR}\lora_adapter_fast"
processor_dir = fr"{OUTPUT_DIR}\processor"
os.makedirs(adapter_dir, exist_ok=True)
model.save_pretrained(adapter_dir)
processor.save_pretrained(processor_dir)
print(f"LoRA адаптер сохранён: {adapter_dir}")
print(f"Processor сохранён:      {processor_dir}")

print("→ Sanity-инференс с безопасными настройками...")

from transformers import pipeline
from peft import PeftModel

base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR, device_map="auto")
ft_model  = PeftModel.from_pretrained(base_model, adapter_dir)
ft_model.eval()

pipe = pipeline(
    task="automatic-speech-recognition",
    model=ft_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    generate_kwargs={
        "task": "transcribe",
        "language": "russian",
        "max_new_tokens": 64,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
        "length_penalty": 1.0,
        "temperature": 0.8,
        "return_timestamps": False,
        "num_beams": 5,
        "do_sample": False,
    },
)

if len(val_ds):
    sample = val_ds[0]["audio"]
    if sample.get("bytes"):
        tmp_path = fr"{OUTPUT_DIR}\_tmp.wav"
        with open(tmp_path, "wb") as f:
            f.write(sample["bytes"])
        test_path = tmp_path
    else:
        test_path = sample["path"]
    res = pipe(test_path)
    print("ASR sample:", json.dumps(res, ensure_ascii=False)[:400], "...")
else:
    print("val_ds пуст — пропускаю sanity-инференс.")