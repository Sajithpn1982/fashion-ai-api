import torch
import numpy as np
import open_clip
from PIL import Image
from config import DEVICE

_model = None
_preprocess = None
_tokenizer = None

def load_model():
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    _model.eval().to(DEVICE)
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")

def normalize(vec: torch.Tensor) -> np.ndarray:
    vec = vec / torch.clamp(vec.norm(dim=-1, keepdim=True), min=1e-6)
    return vec.cpu().numpy().astype("float32")

def encode_text(text: str) -> np.ndarray:
    tokens = _tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        return normalize(_model.encode_text(tokens))

def encode_image(img: Image.Image) -> np.ndarray:
    img = _preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return normalize(_model.encode_image(img))
