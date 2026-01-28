# ColorizedNarrative

## Quickstart:
1. Docker compose up -d
2. Now you can send HTML POST on '0.0.0.0', port=5000:
```python
import requests

url = "http://localhost:5000/asr"

with open("speech.wav", "rb") as f:
    files = {
        "audio": f
    }
    response = requests.post(url, files=files)

print(response.json())
``` 
In response there will be list of base64-encoded images.

OR you can:
1. Docker compose up -d
2. Go to server_name="127.0.0.1", server_port=7860 and use WebGUI

## Промежуточный отчёт команды:

### Алексей Павлов.
Рефакторинг DreamBoothDataset для FLUX
 • Переписал DreamBoothDataset, чтобы не грузить все изображения в память, а читать и аугментировать их лениво в getitem.
 • Добавил поддержку двух режимов: через 🤗 Datasets (dataset_name) и через обычную папку (instance_data_dir).
 • Реализовал поддержку caption_column: если есть подписи в датасете — использую их, иначе — глобальный instance_prompt.
 • Сохранил поддержку prior-preservation (class_data_root, class-изображения).

Коллатор и prior-preservation
 • Настроил collate_fn, который собирает pixel_values и prompts и при --with_prior_preservation добавляет класс-часть в тот же батч.
 • Разные learning rate’ы для FLUX
 • Вынес LoRA-параметры transformer и text_encoder_one в разные параметр-группы для оптимизатора.
 • Задал общий --learning_rate для трансформера и отдельный --text_encoder_lr для текстового энкодера.
 • Настроил оптимизаторы AdamW и prodigy, учёл особенности LR и weight decay для текстового энкодера.
 • Тюнинг текстового энкодера под русский
 • Включил обучение текстового энкодера через флаг --train_text_encoder.
 • Добавил отдельный lr для текстового энкодера, чтобы аккуратно адаптировать его под русские промпты.
 • Правки в train_text_to_image_lora_sdxl.py (SDXL LoRA)
 • Добавил аргумент --text_encoder_lr в скрипт SDXL.
 • Сделал логику: без --text_encoder_lr всё работает как раньше (один LR), с ним — делятся параметры на UNet и text encoder c разными LR.
 • Обёртки bash-скриптов для запуска
 • Написал обёрточные скрипты для FLUX и SDXL, которые:
 • добавляют --train_text_encoder и --text_encoder_lr=...;
 • задают свой output_dir, не ломая базовые скрипты.

 ### Михаил Калинкин.
 1.  Пайплайн извлечения сцены:
 • класс ScenePipeline на Qwen/Qwen2.5-3B-Instruct (AutoModelForCausalLM/AutoTokenizer, device_map="auto", детерминированная генерация без сэмплинга),
 • единая обёртка llm() для всех LLM-вызовов,
 • normalize_text: нормализация русских реплик (пунктуация и опечатки)
 • extract_scene: извлечение чисто визуального описания по тексту персонажа, строгий JSON-формат {"scene", "style", "details"} 
 • build_prompt: фильтрация и нормализация details (1–5 слов, максимум 4 детали), сбор финального промпта под T2I-модель (scene + details + style + "детализированная иллюстрация, реалистичный свет, высокое качество"),
 • process: принимает payload с диаризованными segments, по каждому сегменту делает LLM-нормализацию текста, агрегирует реплики по speaker, для каждого спикера строит scene/style/details и visual_prompt, на выходе возвращает обновлённые segments с normalized_text и список speakers с полями speaker/text/scene/style/details/visual_prompt.
 
 2. Демо-ноутбук:
 • scene_pipeline_demo.ipynb: загрузка структурированных диаризованных JSON из data/diarized_samples, инициализация ScenePipeline,
 • три этапа вывода:
 • исходные сегменты (id/start/end/speaker/text),
 • сегменты после нормализации 
 • агрегированные сцены по спикерам (полный текст спикера, scene/style/details, готовый visual_prompt под генерацию изображений).

### Андрей Зотов:
1. Обучающий пайплайн:

• чистый и детерминированный сплит SOVA-датасета,
• текстовая нормализация под русский,
• фильтрация битых/коротких аудио,
• кастомный коллатор, приводящий аудио к нужному формату Whisper (80×3000),
• LoRA-обёртка для Whisper-small (обучаются Q/V-проекции),
• патч forward PEFT под input_features,
• стабильный TrainingArguments с FP16/BF16 и прогревом,
• обучение до N шагов и сохранение адаптера + процессора,
• sanity-инференс на val_ds.

2. Инференс-пайплайн:

• загрузка сохранённого процессора и LoRA-адаптера,
• тонкая предобработка аудио,
• утилиты для проверки качества на val_ds.

3. Диаризация + ASR:

• интегрирован pyannote speaker-diarization-3.1, работающий на GPU,
• функция diarize_and_transcribe, которая:
-принимает аудио и путь к wav,
-делает диаризацию и ASR по каждому сегменту,
-возвращает список сегментов с полями speaker/start/end/text/duration/max_new_tokens.
• сценарий для реального файла (videoplayback.m4a, 20с),
• экспорт диаризованного транскрипта в структурированный JSON.

### Михаил Павлов:
1. Скрапинг Интернета и компановка датасетов Audio-Text и Text-Image. Датасеты были очищены от мусора, нормированны и залиты на HuggingFace
2. Создание класса для единого итерирования по датасету, вне зависимости от модальности.
3. Курирование команды, планировка архитектуры.
4. Сделать систему выгрузки и загрузки разных частей модели через vLLM.
5. Написать бэкэнд на фласке для доступа к модели.
6. Создать минимальынй gradio интерфейс (под вопросом).
4. Написать docker compose, протестировать пайплайн в стресс-тестах.

## Итоговый проект:
Итоговый пайплайн модели следующий: Аудио на вход -> Текст, полученный с помощью диаризации -> обработка текста, приведение его в вид промпта для text2image модели, перевод на английский -> получение итоговой картинки из промпта.
