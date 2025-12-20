from typing import Any, Dict, List
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ScenePipeline:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", max_new_tokens: int = 256, max_prompt_len : int = 77, device: str = "auto"):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_prompt_len = max_prompt_len
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype="auto",
            trust_remote_code=True
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

    def llm(self, user_text: str):
        messages = [{"role": "user", "content": user_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text

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
            '  \"scene\": \"короткое нейтральное описание сцены одним предложением, максимум 15 слов\",\n'
            '  \"style\": \"краткое описание стиля, 2-4 слова\",\n'
            '  \"details\": [\"1-5 слов, конкретная визуальная деталь 1\", \"деталь 2\", \"...\"]\n'
            "}\n"
            "Не добавляй ничего вне JSON.\n\n"
            f"Текст персонажа: {speaker_text}"
        )
        raw = self.llm(prompt)
        data = json.loads(raw)
        return data

    def translate_to_en(self, prompt_ru: str):
        prompt = (
            "Translate the following text from Russian to English.\n"
            "Preserve the exact meaning and the order of all elements.\n"
            "Do not add new details, do not remove existing ones.\n"
            "Do not rephrase or rewrite stylistically.\n"
            "Transliterate proper names into Latin characters.\n\n"
            "Return strictly a JSON object of the form:\n"
            "{\n"
            '  \"en\": \"translated text in one line\"\n'
            "}\n"
            "Do not add anything outside the JSON object.\n\n"
            f"Text: {prompt_ru}"
        )
        raw = self.llm(prompt)
        data = json.loads(raw)
        return str(data.get("en", "")).strip()

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
            '  \"en\": \"shortened prompt\"\n'
            "}\n"
            "Do not add anything outside the JSON.\n\n"
            f"Max characters: {max_len}\n"
            f"Text: {prompt_en}"
        )
        raw = self.llm(prompt)
        data = json.loads(raw)
        out = str(data.get("en", "")).strip()
        if len(out) <= max_len:
            return out
        cut = out[:max_len].rsplit(" ", 1)[0]
        cut = cut.rstrip(" ,.;:!?)\"]}'")
        return cut

        
    def build_prompt(self, scene_obj: Dict[str, Any]):
        scene = str(scene_obj.get("scene", "")).strip()
        style = str(scene_obj.get("style", "")).strip()
        details_raw = scene_obj.get("details") or []
        if not isinstance(details_raw, list):
            details_raw = [details_raw]
        filtered_details: List[str] = []
        for d in details_raw:
            d = str(d).strip()
            if not d:
                continue
            n_words = len(d.split())
            if n_words < 1 or n_words > 5:
                continue
            filtered_details.append(d)
        filtered_details = filtered_details[:4]
        prompt = (
            "Собери промпт для генерации изображения (text-to-image) на основе полей scene/style/details.\n"
            "Промпт должен быть одной строкой.\n\n"
            "Верни строго JSON вида:\n"
            "{\n"
            '  \"visual_prompt\": \"одна строка, готовый t2i промпт\"\n'
            "}\n"
            "Не добавляй ничего вне JSON.\n\n"
            f"scene: {scene}\n"
            f"style: {style}\n"
            f"details: {json.dumps(filtered_details, ensure_ascii=False)}\n\n"
        )
        raw = self.llm(prompt)
        data = json.loads(raw)
        prompt_ru = str(data.get("visual_prompt", "")).strip()
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
        speaker_full_texts = {
            spk: " ".join(parts) for spk, parts in speaker_texts.items()
        }
        speakers_out = []
        for spk, full_text in speaker_full_texts.items():
            scene_obj = self.extract_scene(full_text)
            visual_prompt = self.build_prompt(scene_obj)
            speakers_out.append(
                {
                    "speaker": spk,
                    "text": full_text,
                    "scene": scene_obj,
                    "visual_prompt": visual_prompt,
                }
            )
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
