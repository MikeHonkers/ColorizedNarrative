#!/usr/bin/env python3

from flask import Flask, request, jsonify
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
import requests
import torch

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "../models/qwen")
T2I_URL = os.getenv("T2I_URL", "http://localhost:5002/t2i")
MAX_NEW_TOKENS = 256
MAX_PROMPT_LEN = 77

local_has_pytorch = os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")) or \
                    os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin"))
model_id = MODEL_PATH if local_has_pytorch else "Qwen/Qwen2.5-3B-Instruct"

print(f"[Connector] Loading {model_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH if os.path.exists(os.path.join(MODEL_PATH, "tokenizer_config.json")) else "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True
)

print("[Connector] Ready")


def llm(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


class ScenePipeline:
    def __init__(self, max_prompt_len: int = MAX_PROMPT_LEN):
        self.max_prompt_len = max_prompt_len

    def normalize_text(self, text: str) -> str:
        prompt = (
            "Исправь пунктуацию и очевидные ошибки в тексте."
            "Не добавляй пояснений и ничего лишнего. "
            "Верни только исправленный текст одной строкой.\n"
            f"{text}"
        )
        return llm(prompt)

    def extract_scene(self, speaker_text: str) -> Dict[str, Any]:
        prompt = (
            "Проанализируй текст персонажа и опиши визуальную сцену для генерации изображения.\n"
            "Опирайся только на то, что можно увидеть глазами: люди, объекты, фон, позы, "
            "одежда, освещение, окружение. Не используй метафоры, эмоции, философию и мысли.\n"
            "Не добавляй в сцену людей или объекты, которых явно нет в тексте персонажа.\n\n"
            "Верни строго JSON вида:\n"
            "{\n"
            '  "scene": "короткое нейтральное описание сцены одним предложением, максимум 15 слов",\n'
            '  "style": "краткое описание стиля, 2-4 слова",\n'
            '  "details": ["1-5 слов, конкретная визуальная деталь 1", "деталь 2", "..."]\n'
            "}\n"
            "Не добавляй ничего вне JSON.\n\n"
            f"Текст персонажа: {speaker_text}"
        )
        raw = llm(prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"scene": "", "style": "", "details": []}

    def translate_to_en(self, prompt_ru: str) -> str:
        prompt = (
            "Translate the following text from Russian to English.\n"
            "Preserve the exact meaning and the order of all elements.\n"
            "Do not add new details, do not remove existing ones.\n"
            "Return strictly a JSON object of the form:\n"
            '{\n  "en": "translated text in one line"\n}\n'
            "Do not add anything outside the JSON object.\n\n"
            f"Text: {prompt_ru}"
        )
        raw = llm(prompt)
        try:
            data = json.loads(raw)
            return str(data.get("en", "")).strip()
        except:
            return prompt_ru

    def shorten_prompt_en(self, prompt_en: str, max_len: int = 77) -> str:
        if len(prompt_en) <= max_len:
            return prompt_en
        prompt = (
            f"Shorten the following text-to-image prompt to be at most {max_len} characters.\n"
            "Keep concrete nouns and actions.\n"
            'Return strictly a JSON object: {{"en": "shortened prompt"}}\n\n'
            f"Text: {prompt_en}"
        )
        raw = llm(prompt)
        try:
            data = json.loads(raw)
            out = str(data.get("en", "")).strip()
            if len(out) <= max_len:
                return out
            return out[:max_len].rsplit(" ", 1)[0].rstrip(" ,.;:!?")
        except:
            return prompt_en[:max_len]

    def build_prompt(self, scene_obj: Dict[str, Any]) -> str:
        scene = str(scene_obj.get("scene", "")).strip()
        style = str(scene_obj.get("style", "")).strip()
        details_raw = scene_obj.get("details") or []
        if not isinstance(details_raw, list):
            details_raw = [details_raw]
        filtered = [str(d).strip() for d in details_raw if 1 <= len(str(d).split()) <= 5][:4]
        
        prompt = (
            "Собери промпт для генерации изображения на основе полей scene/style/details.\n"
            'Верни строго JSON вида:\n{{"visual_prompt": "одна строка"}}\n\n'
            f"scene: {scene}\nstyle: {style}\ndetails: {json.dumps(filtered, ensure_ascii=False)}"
        )
        raw = llm(prompt)
        try:
            data = json.loads(raw)
            prompt_ru = str(data.get("visual_prompt", "")).strip()
        except:
            prompt_ru = f"{scene}, {style}, {', '.join(filtered)}"
        
        prompt_en = self.translate_to_en(prompt_ru)
        return self.shorten_prompt_en(prompt_en, max_len=self.max_prompt_len)

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        segments = payload.get("segments", [])
        normalized_texts = []
        speaker_texts = {}
        
        for seg in segments:
            raw_text = seg.get("text") or ""
            norm = self.normalize_text(raw_text)
            normalized_texts.append(norm)
            spk = seg.get("speaker")
            if spk not in speaker_texts:
                speaker_texts[spk] = []
            speaker_texts[spk].append(norm)
        
        speakers_out = []
        for spk, parts in speaker_texts.items():
            full_text = " ".join(parts)
            scene_obj = self.extract_scene(full_text)
            visual_prompt = self.build_prompt(scene_obj)
            speakers_out.append({
                "speaker": spk,
                "text": full_text,
                "scene": scene_obj,
                "visual_prompt": visual_prompt,
            })
        
        out_segments = []
        for seg, norm in zip(segments, normalized_texts):
            new_seg = dict(seg)
            new_seg["normalized_text"] = norm
            out_segments.append(new_seg)
        
        return {
            "schema_version": payload.get("schema_version", 1),
            "language": payload.get("language", "ru"),
            "segments": out_segments,
            "speakers": speakers_out,
        }


scene_pipeline = ScenePipeline()


@app.route('/connector', methods=['POST'])
def connector():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        payload = request.get_json()
        print(f"[Connector] Processing {len(payload.get('segments', []))} segments")
        
        result = scene_pipeline.process(payload)
        print(f"[Connector] Generated {len(result.get('speakers', []))} prompts")
        
        print("[Connector] Sending to T2I")
        try:
            response = requests.post(T2I_URL, json=result, timeout=300)
            response.raise_for_status()
            return jsonify(response.json()), 200
        except requests.exceptions.RequestException as e:
            print(f"[Connector] T2I error: {e}")
            return jsonify({"error": "T2I failed", "intermediate_result": result}), 502
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "connector"}), 200


if __name__ == '__main__':
    print("[Connector] http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
