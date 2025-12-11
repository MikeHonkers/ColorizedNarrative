from datasets import load_dataset
from src.config import MODELS_CONFIG, get_path, ensure_dir


def download_dataset():
    dataset_name = MODELS_CONFIG['dataset']['name']
    local_path = MODELS_CONFIG['dataset']['local_path']
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    output_path = ensure_dir(local_path)
    
    print(f"Saving dataset to {output_path}")
    dataset.save_to_disk(str(output_path))
    
    print(f"Dataset saved successfully!")
    print(f"Train split size: {len(dataset['train'])}")
    
    return dataset


if __name__ == "__main__":
    download_dataset()

