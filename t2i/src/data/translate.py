import json
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.config import MODELS_CONFIG, get_path


def translate_text(model, tokenizer, text_ru):
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
        f"Text: {text_ru}"
    )
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    try:
        result = json.loads(response.strip())
        return result.get("en", text_ru)
    except json.JSONDecodeError:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            try:
                result = json.loads(response[start:end])
                return result.get("en", text_ru)
            except:
                pass
        return text_ru


def translate_dataset():
    local_path = MODELS_CONFIG['dataset']['local_path']
    caption_col = MODELS_CONFIG['dataset']['caption_column']
    caption_en_col = MODELS_CONFIG['dataset']['caption_column_en']
    
    dataset_path = get_path(local_path)
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    
    print("Loading Qwen2.5-3B-Instruct...")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    def translate_example(example):
        translated = translate_text(model, tokenizer, example[caption_col])
        return {caption_en_col: translated}
    
    print(f"Translating {caption_col} -> {caption_en_col}")
    dataset = dataset.map(
        translate_example,
        desc="Translating"
    )
    
    print(f"Saving dataset to {dataset_path}")
    dataset.save_to_disk(str(dataset_path))
    
    print("Done!")
    print(f"Sample: {dataset['train'][0][caption_col]} -> {dataset['train'][0][caption_en_col]}")
    
    return dataset


if __name__ == "__main__":
    translate_dataset()
