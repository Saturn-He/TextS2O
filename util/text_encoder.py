from pathlib import Path

import torch
from PIL import Image


class ClipTextEncoder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", text_dim=768):
        self.model_name = model_name
        self.text_dim = text_dim

    def encode_texts(self, texts):
        """
        Placeholder for CLIP-ViT-L/14 text encoding.
        Replace this stub with actual encoding logic.
        """
        return torch.zeros(len(texts), self.text_dim)


class QwenVLTextGenerator:
    def __init__(self, model_name="Qwen2-VL-72B"):
        self.model_name = model_name

    def generate_text_for_image(self, image):
        """
        Placeholder for Qwen2-VL-72B API call.
        Replace this stub with actual API interaction.
        """
        return ""


def _text_path_for_sar(sar_path, output_dir):
    sar_path = Path(sar_path)
    output_dir = Path(output_dir)
    return output_dir / f"{sar_path.stem}.txt"


def load_texts_for_pairs(sar_paths, output_dir):
    texts = []
    for sar_path in sar_paths:
        text_path = _text_path_for_sar(sar_path, output_dir)
        if not text_path.exists():
            raise FileNotFoundError(f"Missing text file: {text_path}")
        texts.append(text_path.read_text(encoding="utf-8").strip())
    return texts


def load_texts_for_names(names, output_dir):
    output_dir = Path(output_dir)
    texts = []
    for name in names:
        text_path = output_dir / f"{Path(name).stem}.txt"
        if not text_path.exists():
            raise FileNotFoundError(f"Missing text file: {text_path}")
        texts.append(text_path.read_text(encoding="utf-8").strip())
    return texts


def ensure_text_data(sar_paths, opt_paths, output_dir, llm_model_name="Qwen2-VL-72B"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = QwenVLTextGenerator(model_name=llm_model_name)
    for sar_path, opt_path in zip(sar_paths, opt_paths):
        text_path = _text_path_for_sar(sar_path, output_dir)
        if text_path.exists():
            continue
        opt_image = Image.open(opt_path).convert("RGB")
        text = generator.generate_text_for_image(opt_image)
        text_path.write_text(text, encoding="utf-8")
