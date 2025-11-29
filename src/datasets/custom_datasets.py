import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms

class TextToAudioDataset(Dataset):
    def __init__(self, hf_path="MikeHonkers/SOVA-audiobooks-100k", sample_rate=16000, max_audio_len_sec=10): 
        self.dataset = load_dataset(hf_path, split="train")
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_audio_len_sec)
        self.texts = self.dataset["text"]
        self.audios = self.dataset["audio"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.texts[idx]
        audio = self.audios[idx].get_all_samples().data

        if audio.size(1) > self.max_samples:
            audio = audio[:, :self.max_samples]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.max_samples - audio.size(1)))

        return {
            "input_text": text,
            "target_audio": audio,
        }
        
    def collate_fn(self, batch):
        texts = [item["input_text"] for item in batch]
        audios = torch.stack([item["target_audio"] for item in batch])
        return {"input_text": texts, "target_audio": audios}

class TextToImageDataset(Dataset):
    def __init__(self, hf_path="MikeHonkers/Images-ru_cap-13k", image_size=(256, 256), format="RGB"):
        self.dataset = load_dataset(hf_path, split="train")
        self.image_size = image_size
        self.format = format

        self.texts = self.dataset["caption"]
        self.images = self.dataset["image"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.images[idx]
        image = image.resize((256, 256))
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        return {
            "input_text": text,
            "target_image": image
        }

    def collate_fn(self, batch):
        texts = [item["input_text"] for item in batch]
        images = torch.stack([item["target_image"] for item in batch])
        return {"input_text": texts, "target_image": images}