"""HEAVY: foundation-model forward pass, disk-cached by image id. Run on Colab."""
import torch
from PIL import Image
from tqdm import tqdm

from src.data.imagenet_dog15 import ImageRecord
from src.device import get_device
from src.embeddings.cache import has_cached, save_embedding


def load_clip_vit_b32(device: torch.device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def extract_clip_embeddings(
    records: list[ImageRecord],
    dataset_name: str,
    backbone_name: str = "clip_vit_b32",
    device: torch.device | None = None,
    batch_size: int = 64,
) -> None:
    """Extract + cache CLIP embeddings for records not already cached."""
    device = device or get_device()
    todo = [r for r in records if not has_cached(dataset_name, backbone_name, r.image_id)]
    if not todo:
        print(f"[extract] all {len(records)} embeddings already cached for {backbone_name}")
        return

    print(f"[extract] {len(todo)}/{len(records)} embeddings to compute on {device}")
    model, preprocess = load_clip_vit_b32(device)

    for i in tqdm(range(0, len(todo), batch_size), desc="extracting"):
        batch = todo[i : i + batch_size]
        imgs = torch.stack([preprocess(Image.open(r.path).convert("RGB")) for r in batch]).to(device)
        feats = model.encode_image(imgs)
        feats = feats.cpu().numpy()
        for r, vec in zip(batch, feats):
            save_embedding(dataset_name, backbone_name, r.image_id, vec)
