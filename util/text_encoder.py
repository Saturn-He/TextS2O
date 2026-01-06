import base64
import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional

import requests
import torch
from PIL import Image
from torchvision.transforms import functional as F
from transformers import CLIPTextModel, CLIPTokenizer


class ClipTextEncoder:
    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        text_dim=768,
        device="cpu",
        batch_size=32,
    ):
        self.model_name = model_name
        self.text_dim = text_dim
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Iterable[str],
        device: Optional[torch.device] = None,
        output_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        texts = list(texts)
        if not texts:
            return torch.empty(0, self.text_dim)
        device = torch.device(device) if device is not None else self.device
        output_device = torch.device(output_device) if output_device is not None else torch.device("cpu")
        if device != self.device:
            self.model.to(device)
            self.device = device
        features: List[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            pooled = outputs.pooler_output
            if pooled.shape[-1] != self.text_dim:
                raise ValueError(
                    f"CLIP text dim mismatch: expected {self.text_dim}, got {pooled.shape[-1]}"
                )
            features.append(pooled.detach().to(output_device))
        return torch.cat(features, dim=0)


SYSTEM_PROMPT_EN = """System Role: You are a senior expert in Synthetic Aperture Radar (SAR) remote sensing interpretation. Your task is to analyze optical images and translate them into professional technical descriptions of corresponding SAR backscattering characteristics.

Please provide a concise description within 40-60 words.

Task Logic: 
1. Analyze the physical properties of terrain and objects in the optical image (material, surface roughness, geometric structure).
2. Translate these properties into radar scattering mechanisms.

Strict Constraints:
- NO COLORS: Strictly forbid mentioning any color information (red, green, blue, etc.).
- NO ORIENTATIONS: Strictly forbid directional terms like left, right, top, bottom, etc. Use relative spatial relationships like 'adjacent to', 'surrounded by', or 'interspersed with'.
- NO OPTICAL ARTIFACTS: Strictly forbid mentioning sunlight, clouds, or lens flares.

Required Vocabulary:
- Scattering Mechanisms: Specular reflection, Volume scattering, Double-bounce scattering, Surface roughness, Bragg scattering.
- Material Attributes: High/Low dielectric constant, Metallic, Non-metallic, Moisture content, Concrete, Vegetation density.
- Structural Patterns: Regular grid, Cluttered texture, Homogeneous area, Discrete point-like targets, Corner reflectors."""


class QwenVLTextGenerator:
    def __init__(
        self,
        model_name="Qwen2-VL-72B",
        api_key: Optional[str] = sk-e65c52e75ee74698b15da25033d1c2c1,
        api_base: Optional[str] = None,
        prompt: str = SYSTEM_PROMPT_EN,
        timeout: int = 60,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("QWEN_VL_API_KEY")
        self.api_base = api_base or os.environ.get("QWEN_VL_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.prompt = prompt
        self.timeout = timeout

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate_text_for_image(self, image: Image.Image) -> str:
        if not self.api_key:
            raise ValueError("Missing QWEN_VL_API_KEY for Qwen2-VL-72B API access.")
        encoded = self._image_to_base64(image)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": 256,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_base}/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = "".join(chunk.get("text", "") for chunk in message)
        return str(message).strip()


class TextConditioner:
    def __init__(self, text_encoder: ClipTextEncoder, text_data_dir: Optional[Path] = None,
                 cfg_scale: Optional[float] = None):
        self.text_encoder = text_encoder
        self.text_data_dir = Path(text_data_dir) if text_data_dir is not None else None
        self.cfg_scale = cfg_scale

    def get_text_features(self, names: Iterable[str], device: torch.device) -> Optional[torch.Tensor]:
        if self.text_data_dir is None:
            return None
        texts = load_texts_for_names(names, self.text_data_dir)
        text_features = self.text_encoder.encode_texts(texts, device=device, output_device=device)
        return text_features.to(device=device, dtype=torch.float32)

    @contextmanager
    def cfg_scale_context(self, model) -> Iterable[None]:
        if self.cfg_scale is None:
            yield
            return
        original = model.cfg_scale
        model.cfg_scale = self.cfg_scale
        try:
            yield
        finally:
            model.cfg_scale = original


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


def _ensure_pil_image(image):
    if isinstance(image, Image.Image):
        return image
    return F.to_pil_image(image)


def ensure_text_data(
    sar_paths,
    opt_paths,
    output_dir,
    llm_model_name="Qwen2-VL-72B",
    prompt=SYSTEM_PROMPT_EN,
    transform=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = QwenVLTextGenerator(model_name=llm_model_name, prompt=prompt)
    for sar_path, opt_path in zip(sar_paths, opt_paths):
        text_path = _text_path_for_sar(sar_path, output_dir)
        if text_path.exists():
            continue
        opt_image = Image.open(opt_path).convert("RGB")
        if transform is not None:
            sar_image = Image.open(sar_path).convert("L")
            try:
                _sar_image, opt_image = transform(sar_image, opt_image, Path(sar_path).name)
            except TypeError:
                _sar_image, opt_image = transform(sar_image, opt_image)
            opt_image = _ensure_pil_image(opt_image)
        text = generator.generate_text_for_image(opt_image)
        text_path.write_text(text, encoding="utf-8")
