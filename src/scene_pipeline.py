from typing import Any, Dict, List
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ScenePipeline:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", max_new_tokens: int = 256):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
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
            '  \"scene\": \"короткое нейтральное описание сцены одним предложением, максимум 15 слов, только то, что видно\",\n'
            '  \"style\": \"краткое описание стиля, 2-4 слова (например: кинематографичный реализм, цифровая иллюстрация)\",\n'
            '  \"details\": [\"1-5 слов, конкретная визуальная деталь 1\", \"деталь 2\", \"...\"]\n'
            "}\n"
            "Не добавляй ничего вне JSON.\n\n"
            f"Текст персонажа: {speaker_text}"
        )

        raw = self.llm(prompt)
        data = json.loads(raw)
        return data

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

        parts = []
        if scene:
            parts.append(scene)
        if filtered_details:
            parts.append(", ".join(filtered_details))
        if style:
            parts.append(style)

        parts.append("детализированная иллюстрация, реалистичный свет, высокое качество")
        prompt = ", ".join(parts)
        return prompt

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        lang = payload.get("language", "ru")
        segments = payload.get("segments", [])

        normalized_texts = []
        speaker_texts: Dict[str, List[str]] = {}

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
