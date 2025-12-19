from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable
import yaml
from tqdm import tqdm
from src.scene_pipeline import ScenePipeline


CONFIG_PATH = Path("config/scene.yaml")


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_inputs(p: Path) -> Iterable[Path]:
    if p.is_file():
        yield p
        return
    yield from sorted(p.glob("*.json"))


def main():
    cfg = load_yaml(CONFIG_PATH)
    model_id = str(cfg.get("model_id", "Qwen/Qwen2.5-3B-Instruct"))
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    max_prompt_len = int(cfg.get("max_prompt_len", 77))
    device = str(cfg.get("device", "mps"))
    inp = Path(cfg.get("input_path", "data/diarized_samples"))
    outp = Path(cfg.get("output_path", "data/scene_outputs"))
    outp.mkdir(parents=True, exist_ok=True)
    limit = int(cfg.get("limit", 0) or 0)
    pipe = ScenePipeline(model_id=model_id, max_new_tokens=max_new_tokens, max_prompt_len=max_prompt_len, device=device)
    files = list(iter_inputs(inp))
    if limit:
        files = files[:limit]
    for fp in tqdm(files, desc="scene"):
        with fp.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        result = pipe.process(payload)
        out_file = outp / fp.name.replace(".json", "_scene.json")
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Done. Wrote {len(files)} file(s) into: {outp}")


if __name__ == "__main__":
    main()
