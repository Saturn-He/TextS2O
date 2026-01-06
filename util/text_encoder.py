import torch


class ClipTextEncoder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", text_dim=768):
        self.model_name = model_name
        self.text_dim = text_dim

    def fetch_texts_for_pairs(self, sar_paths, opt_paths):
        """
        Placeholder for external large-model API call.
        Replace this stub with real text retrieval logic.
        """
        return [""] * len(sar_paths)

    def encode_texts(self, texts):
        """
        Placeholder for CLIP-ViT-L/14 text encoding.
        Replace this stub with actual encoding logic.
        """
        return torch.zeros(len(texts), self.text_dim)
