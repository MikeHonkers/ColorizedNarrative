from flask import Flask, request, jsonify
import os
import json
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
from typing import Dict, Any
import requests

app = Flask(__name__)

TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")
T2I_URL = "http://t2i:5002/t2i"
QWEN_MODEL_NAME = "qwen_onnx"
MAX_NEW_TOKENS = 256
MAX_PROMPT_LEN = 77

client = httpclient.InferenceServerClient(url=TRITON_URL)

tokenizer = AutoTokenizer.from_pretrained("/models/qwen", trust_remote_code=True)

def load_model(model_name: str):
    if not client.is_model_ready(model_name):
        print(f"Loading model: {model_name}")
        client.load_model(model_name)

def unload_model(model_name: str):
    if client.is_model_ready(model_name):
        print(f"Unloading model: {model_name}")
        client.unload_model(model_name)

class QwenTritonGenerator:
    def __init__(self, model_name: str = QWEN_MODEL_NAME, max_new_tokens: int = MAX_NEW_TOKENS):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(text)

        generated_ids = []
        current_input_ids = np.array([input_ids], dtype=np.int64)

        for _ in range(self.max_new_tokens):
            infer_input = httpclient.InferInput("input_ids", current_input_ids.shape, "INT64")
            infer_input.set_data_from_numpy(current_input_ids)

            result = client.infer(
                model_name=self.model_name,
                inputs=[infer_input],
                outputs=[httpclient.InferRequestedOutput("logits")]
            )
            logits = result.as_numpy("logits")

            next_token = int(np.argmax(logits[0, -1, :]))
            generated_ids.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

            current_input_ids = np.array([[next_token]], dtype=np.int64)

        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

generator = QwenTritonGenerator()

class ScenePipeline:
    def __init__(self, max_new_tokens: int = 256, max_prompt_len: int = 77):
        self.max_new_tokens = max_new_tokens
        self.max_prompt_len = max_prompt_len

    def llm(self, user_text: str):
        return generator.generate(user_text)

    def normalize_text(self, text: str):
        prompt = (
            "Исправь пунктуацию и очевидные ошибки в тексте."
            "Не добавляй пояснений и ничего лишнего. "
            "Верни только исправленный текст одной строкой.\n"
            f"{text}"
        )
        return self.llm(prompt)

    def extract_scene(self, speaker_text: str):
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
        raw = self.llm(prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"scene": "", "style": "", "details": []}

    def translate_to_en(self, prompt_ru: str):
        prompt = (
            "Translate the following text from Russian to English.\n"
            "Preserve the exact meaning and the order of all elements.\n"
            "Do not add new details, do not remove existing ones.\n"
            "Do not rephrase or rewrite stylistically.\n"
            "Transliterate proper names into Latin characters.\n\n"
            "Return strictly a JSON object of the form:\n"
            "{\n"
            '  "en": "translated text in one line"\n'
            "}\n"
            "Do not add anything outside the JSON object.\n\n"
            f"Text: {prompt_ru}"
        )
        raw = self.llm(prompt)
        try:
            data = json.loads(raw)
            return str(data.get("en", "")).strip()
        except:
            return prompt_ru

    def shorten_prompt_en(self, prompt_en: str, max_len: int = 77):
        if len(prompt_en) <= max_len:
            return prompt_en
        prompt = (
            "Shorten the following text-to-image prompt to be at most the character limit.\n"
            f"Hard rule: len(en) <= {max_len}. No exceptions.\n"
            "Keep concrete nouns and actions.\n"
            "Prefer dropping last comma-separated segments until it fits.\n"
            "Avoid ending with unfinished connectors.\n"
            "Return strictly a JSON object exactly of the form:\n"
            "{\n"
            '  "en": "shortened prompt"\n'
            "}\n"
            "Do not add anything outside the JSON.\n\n"
            f"Max characters: {max_len}\n"
            f"Text: {prompt_en}"
        )
        raw = self.llm(prompt)
        try:
            data = json.loads(raw)
            out = str(data.get("en", "")).strip()
            if len(out) <= max_len:
                return out
            return out[:max_len].rsplit(" ", 1)[0].rstrip(" ,.;:!?)\"]}'")
        except:
            return prompt_en[:max_len]

    def build_prompt(self, scene_obj: Dict[str, Any]):
        scene = str(scene_obj.get("scene", "")).strip()
        style = str(scene_obj.get("style", "")).strip()
        details_raw = scene_obj.get("details") or []
        if not isinstance(details_raw, list):
            details_raw = [details_raw]
        filtered_details = [str(d).strip() for d in details_raw if 1 <= len(str(d).split()) <= 5][:4]
        prompt = (
            "Собери промпт для генерации изображения (text-to-image) на основе полей scene/style/details.\n"
            "Промпт должен быть одной строкой.\n\n"
            "Верни строго JSON вида:\n"
            "{\n"
            '  "visual_prompt": "одна строка, готовый t2i промпт"\n'
            "}\n"
            "Не добавляй ничего вне JSON.\n\n"
            f"scene: {scene}\n"
            f"style: {style}\n"
            f"details: {json.dumps(filtered_details, ensure_ascii=False)}\n\n"
        )
        raw = self.llm(prompt)
        try:
            data = json.loads(raw)
            prompt_ru = str(data.get("visual_prompt", "")).strip()
        except:
            prompt_ru = f"{scene}, {style}, {', '.join(filtered_details)}"
        prompt_en = self.translate_to_en(prompt_ru)
        prompt_en = self.shorten_prompt_en(prompt_en, max_len=self.max_prompt_len)
        return prompt_en

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        lang = payload.get("language", "ru")
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
        speaker_full_texts = {spk: " ".join(parts) for spk, parts in speaker_texts.items()}
        speakers_out = []
        for spk, full_text in speaker_full_texts.items():
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
            "language": lang,
            "segments": out_segments,
            "speakers": speakers_out,
        }

scene_pipeline = ScenePipeline(max_new_tokens=MAX_NEW_TOKENS, max_prompt_len=MAX_PROMPT_LEN)

@app.route('/connector', methods=['POST'])
def connector():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        load_model(QWEN_MODEL_NAME)

        payload = request.get_json()
        result = scene_pipeline.process(payload)
        
        unload_model(QWEN_MODEL_NAME)

        try:
            response = requests.post(
                T2I_URL,
                json=result,
                timeout=120
            )
            response.raise_for_status()
            final_result = response.json()

        except requests.exceptions.RequestException as e:
            print(f"Connector error: {e}")
            return jsonify({
                "error": "Scene generation failed",
                "intermediate_result": result,
                "connector_error": str(e)
            }), 502

        return jsonify(final_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)