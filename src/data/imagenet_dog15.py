"""ImageNet-Dog-15 loader.

15 dog-breed synsets, the standard subset used in deep-clustering literature
(JULE / DAC / DEC-style papers, and reused by the paper this experiment extends).
"""
import os
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DOG15_SYNSETS = {
    "n02085936": "Maltese_dog",
    "n02086646": "Blenheim_spaniel",
    "n02088238": "basset",
    "n02091467": "Norwegian_elkhound",
    "n02097130": "giant_schnauzer",
    "n02099601": "golden_retriever",
    "n02101388": "Brittany_spaniel",
    "n02101556": "clumber",
    "n02102177": "Welsh_springer_spaniel",
    "n02105056": "groenendael",
    "n02105412": "kelpie",
    "n02105855": "Shetland_sheepdog",
    "n02107142": "Doberman",
    "n02110958": "pug",
    "n02112137": "chow",
}
SYNSET_TO_LABEL = {syn: i for i, syn in enumerate(sorted(DOG15_SYNSETS))}

# Matches the sibling `colab_transfer_disentangler.ipynb` notebook's Drive layout
# (save_dir/.cache/<name>/<label>/<%05d>.jpg from its download_domain()). Reusing
# that cache avoids a second gated-HF download for the same dog-breed images.
DEFAULT_DRIVE_CACHE = "/content/drive/MyDrive/representation-learning/.cache/imagenet-dog-15"


@dataclass
class ImageRecord:
    image_id: str
    path: str
    label: int
    synset: str


def _load_from_precached_dir(root: Path) -> list[ImageRecord]:
    """Generic <root>/<label>/*.{jpg,jpeg} layout, as written by download_domain() in
    colab_transfer_disentangler.ipynb. `label` subdirs may be synset ids (n0XXXXXXX) or
    plain ILSVRC integer label indices — both are handled.
    """
    label_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not label_dirs:
        raise FileNotFoundError(f"no label subdirs under {root}")
    if len(label_dirs) != len(DOG15_SYNSETS):
        raise FileNotFoundError(
            f"expected {len(DOG15_SYNSETS)} label dirs under {root}, found {len(label_dirs)}"
        )

    synset_dirs = [d for d in label_dirs if d.name in DOG15_SYNSETS]
    if len(synset_dirs) == len(label_dirs):
        dir_to_label = {d: SYNSET_TO_LABEL[d.name] for d in synset_dirs}
        dir_to_synset = {d: d.name for d in synset_dirs}
    else:
        # Non-synset dir names (e.g. raw ILSVRC label ints) — assign class indices by
        # sorted dir name, since we have no ground-truth synset mapping from names alone.
        print(
            f"[data] label dirs under {root} aren't synset ids "
            f"({[d.name for d in label_dirs]}); assigning class indices by sort order."
        )
        dir_to_label = {d: i for i, d in enumerate(label_dirs)}
        dir_to_synset = {d: d.name for d in label_dirs}

    records = []
    for d in label_dirs:
        img_paths = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.JPEG"))
        for img_path in img_paths:
            records.append(
                ImageRecord(
                    image_id=f"{dir_to_synset[d]}_{img_path.stem}",
                    path=str(img_path),
                    label=dir_to_label[d],
                    synset=dir_to_synset[d],
                )
            )
    return records


def _load_from_local_imagenet(root: Path) -> list[ImageRecord]:
    """ImageFolder-style local ImageNet: root/<synset>/*.JPEG"""
    records = []
    for synset in sorted(DOG15_SYNSETS):
        synset_dir = root / synset
        if not synset_dir.is_dir():
            raise FileNotFoundError(f"synset dir missing: {synset_dir}")
        for img_path in sorted(synset_dir.glob("*.JPEG")) + sorted(synset_dir.glob("*.jpeg")):
            records.append(
                ImageRecord(
                    image_id=f"{synset}_{img_path.stem}",
                    path=str(img_path),
                    label=SYNSET_TO_LABEL[synset],
                    synset=synset,
                )
            )
    return records


def _load_from_hf() -> list[ImageRecord]:
    """Fallback: HF `imagenet-1k` (gated, requires `huggingface-cli login`), filtered to dog15.

    Slower than a local ImageFolder — only use if IMAGENET_ROOT isn't set.
    """
    from datasets import load_dataset

    print(
        "[data] IMAGENET_ROOT not set or invalid; falling back to HF `imagenet-1k` "
        "streaming dataset filtered to ImageNet-Dog-15. This is a SUBSTITUTION from a "
        "local ImageFolder — flagging per experiment protocol. Requires HF auth."
    )
    ds = load_dataset("imagenet-1k", split="train", streaming=True)
    wanted_synsets = set(DOG15_SYNSETS)
    # imagenet-1k on HF encodes label as an int index into a fixed class list, not the
    # synset id directly — we need the label->synset mapping via ds.features.
    label_names = ds.features["label"].names  # type: ignore[attr-defined]
    idx_to_synset = {}
    for idx, name in enumerate(label_names):
        # HF class names are typically "n0XXXXXXX" or human-readable; handle both.
        for syn in wanted_synsets:
            if name.startswith(syn) or syn in name:
                idx_to_synset[idx] = syn

    records = []
    cache_dir = Path(__file__).resolve().parents[2] / "outputs" / "raw_images" / "dog15_hf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, example in enumerate(ds):
        label_idx = example["label"]
        if label_idx not in idx_to_synset:
            continue
        synset = idx_to_synset[label_idx]
        img: Image.Image = example["image"]
        image_id = f"{synset}_{i}"
        img_path = cache_dir / f"{image_id}.jpeg"
        if not img_path.exists():
            img.convert("RGB").save(img_path)
        records.append(
            ImageRecord(image_id=image_id, path=str(img_path), label=SYNSET_TO_LABEL[synset], synset=synset)
        )
    return records


def load_dog15(seed: int = 0) -> list[ImageRecord]:
    """Load all ImageNet-Dog-15 records available.

    Priority:
    1. env var IMAGENET_DOG15_CACHE, or the default Drive path (DEFAULT_DRIVE_CACHE) if
       it exists — a pre-downloaded <label>/*.jpg cache shared with
       colab_transfer_disentangler.ipynb. Preferred: no re-download, no HF auth needed.
    2. env var IMAGENET_ROOT — a full local ImageNet-1k ImageFolder (root/<synset>/*.JPEG).
    3. HF `imagenet-1k` streaming fallback — gated, requires auth, flagged loudly as a
       substitution since it's the slowest/least-preferred path.
    """
    cache_env = os.environ.get("IMAGENET_DOG15_CACHE")
    cache_path = Path(cache_env) if cache_env else Path(DEFAULT_DRIVE_CACHE)
    if cache_path.is_dir():
        try:
            records = _load_from_precached_dir(cache_path)
            print(f"[data] loaded {len(records)} ImageNet-Dog-15 images from precached dir {cache_path}")
            return records
        except FileNotFoundError as e:
            print(f"[data] precached dir load failed ({e}); trying IMAGENET_ROOT next.")

    root_env = os.environ.get("IMAGENET_ROOT")
    if root_env:
        root = Path(root_env)
        try:
            records = _load_from_local_imagenet(root)
            print(f"[data] loaded {len(records)} ImageNet-Dog-15 images from local IMAGENET_ROOT={root}")
            return records
        except FileNotFoundError as e:
            print(f"[data] local IMAGENET_ROOT load failed ({e}); falling back to HF.")

    records = _load_from_hf()
    print(f"[data] loaded {len(records)} ImageNet-Dog-15 images via HF fallback")
    return records


def split_fit_eval(
    records: list[ImageRecord], m: int, seed: int = 0
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Deterministic split: m images for fitting W + training the student, rest held out for eval."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    if m > len(shuffled):
        raise ValueError(f"requested m={m} but only {len(shuffled)} images available")
    fit_set, eval_set = shuffled[:m], shuffled[m:]
    return fit_set, eval_set
