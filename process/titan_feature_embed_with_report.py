import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from tiffslide import TiffSlide
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

# Hey guys, if you ever want to use this, make sure you edit this to fit your directory structure
TRAINING_ROOT_DIR = Path("/bulk/rapae/BMED6460/wsi_project/training_data/wsitils")

# Probably don't need to touch this
WSI_FOLDER_NAME = Path("images")
CSV_NAME = Path("tiger-til-scores-wsitils.csv")
MASK_FOLDER_NAME = Path("tissue-masks")

OUT_DIR = Path("../tiger_titan_embeddings_fused_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 512
STRIDE = 512
PATCH_SIZE_LV0 = 512
BATCH_SIZE = 32
THUMB_MAX_SIDE = 2048
TISSUE_FRAC_THRESHOLD = 0.10
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

CSV_ID_COLS = "image-id"
CSV_TILS_SCORE = "tils-score"
PATHOLOGY_REPORT = "comment"

# I need to do this or else I fill up my gpu
def open_tissue_mask(mask_path, max_side=4096):

    mask_slide = TiffSlide(str(mask_path))
    w, h = mask_slide.dimensions

    scale = max(w, h) / max_side
    if scale < 1:
        scale = 1.0

    out_w = max(1, int(w / scale))
    out_h = max(1, int(h / scale))

    # Read a thumbnail instead of the full huge mask
    mask_img = mask_slide.get_thumbnail((out_w, out_h)).convert("L")
    mask = np.array(mask_img)

    # Binarize
    return (mask > 0).astype(np.uint8)

def generate_level0_coords(slide: TiffSlide,
                           patch_size_lv0: int,
                           stride_lv0: int,
                           tissue_mask: np.ndarray,
                           tissue_threshold: float = 0.10) -> np.ndarray:

    slide_w, slide_h = slide.dimensions
    mask_h, mask_w = tissue_mask.shape

    xs = np.arange(0, max(1, slide_w - patch_size_lv0 + 1), stride_lv0, dtype=np.int64)
    ys = np.arange(0, max(1, slide_h - patch_size_lv0 + 1), stride_lv0, dtype=np.int64)

    coords = []
    for y in ys:
        for x in xs:
            x0_m = int((x / slide_w) * mask_w)
            y0_m = int((y / slide_h) * mask_h)
            x1_m = int(((x + patch_size_lv0) / slide_w) * mask_w)
            y1_m = int(((y + patch_size_lv0) / slide_h) * mask_h)

            x1_m = max(x0_m + 1, min(x1_m, mask_w))
            y1_m = max(y0_m + 1, min(y1_m, mask_h))

            patch_mask = tissue_mask[y0_m:y1_m, x0_m:x1_m]
            if patch_mask.size == 0:
                continue

            if patch_mask.mean() >= tissue_threshold:
                coords.append([x, y])

    if not coords:
        return np.zeros((0, 2), dtype=np.int64)

    return np.asarray(coords, dtype=np.int64)


def read_patches(slide: TiffSlide, coords: np.ndarray, patch_size: int):
    patches = []
    for x, y in coords:
        patch = slide.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
        patches.append(patch)
    return patches


def conch_encode_batch(conch, images_tensor: torch.Tensor) -> torch.Tensor:

    with torch.inference_mode():
        out = conch(images_tensor)
        if isinstance(out, (tuple, list)):
            feats = out[0]
        elif isinstance(out, dict):
            # adjust if needed
            feats = out.get("image_features", None)
        else:
            feats = out

    return feats

def extract_slide_embedding(slide_path: Path, mask_path: Path, titan, conch, eval_transform) -> tuple[
    Any, Any, ndarray]:
    slide = TiffSlide(str(slide_path))

    tissue_mask = open_tissue_mask(mask_path)

    coords = generate_level0_coords(
        slide=slide,
        patch_size_lv0=PATCH_SIZE,
        stride_lv0=STRIDE,
        tissue_mask=tissue_mask,
        tissue_threshold=TISSUE_FRAC_THRESHOLD,
    )

    if len(coords) == 0:
        raise RuntimeError(f"No tissue patches found for {slide_path.name}")

    all_features = []
    for start in range(0, len(coords), BATCH_SIZE):
        batch_coords = coords[start:start + BATCH_SIZE]
        batch_pil = read_patches(slide, batch_coords, PATCH_SIZE)

        batch_tensor = torch.stack([eval_transform(im) for im in batch_pil], dim=0).to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            batch_feats = conch_encode_batch(conch, batch_tensor)

        batch_feats = batch_feats.detach().float().cpu()
        all_features.append(batch_feats)

    features = torch.cat(all_features, dim=0).to(DEVICE)
    coords_t = torch.from_numpy(coords).to(DEVICE)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")), torch.inference_mode():
        slide_embedding = titan.encode_slide_from_patch_features(
            features,
            coords_t,
            PATCH_SIZE_LV0,
        )

    return (
        slide_embedding.detach().float().cpu().squeeze(),
        features.detach().cpu(),
        coords
    )

def encode_notes(notes, titan, max_length=256):

    if isinstance(notes, str):
        notes = [notes]

    tokenizer = titan.text_encoder.tokenizer
    text_tokens = tokenizer(notes)

    max_tokens = titan.text_encoder.context_length - 1
    text_tokens = text_tokens[:, :max_tokens].to(DEVICE)

    with torch.inference_mode():
        z = titan.text_encoder(text_tokens)

    z = z[0]
    z = z.float()
    z = F.normalize(z, dim=-1)
    return z.cpu()

def main():

    wsi_folder_path = TRAINING_ROOT_DIR / WSI_FOLDER_NAME
    score_csv_path = TRAINING_ROOT_DIR / CSV_NAME
    mask_folder_path = TRAINING_ROOT_DIR / MASK_FOLDER_NAME

    # All this is ripped straight from the github https://github.com/mahmoodlab/TITAN
    # Make sure you apply for the model weights and generate a huggingface API key

    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(DEVICE).eval()

    conch, eval_transform = titan.return_conch()
    conch = conch.to(DEVICE).eval()

    df = pd.read_csv(score_csv_path)

    id_col = CSV_ID_COLS if CSV_ID_COLS in df.columns else None
    y_col = CSV_TILS_SCORE if CSV_TILS_SCORE in df.columns else None
    notes_col = PATHOLOGY_REPORT if PATHOLOGY_REPORT in df.columns else None

    if id_col is None or y_col is None:
        raise RuntimeError(
            f"Could not find CSV columns. Found columns: {list(df.columns)}. "
            "You're gonna have to set id_col and y_col manually bub."
        )

    rows_out = []
    X = []
    y = []

    for _, row in df.iterrows():
        slide_id = str(row[id_col])
        target = float(row[y_col])
        note_text = str(row[notes_col])

        slide_path = wsi_folder_path / f"{slide_id}.tif"
        mask_path = mask_folder_path / f"{slide_id}_tissue.tif"
        print(f"Processing {slide_path.name} ...")

        emb, patch_feats, patch_coords = extract_slide_embedding(slide_path, mask_path, titan, conch, eval_transform)
        note_emb = encode_notes(note_text, titan).squeeze(0)

        cos = F.cosine_similarity(
            emb.unsqueeze(0),
            note_emb.unsqueeze(0),
            dim=-1
        )

        fused = torch.cat([emb, note_emb, cos], dim=0)

        np.save(OUT_DIR / f"{slide_id}.npy", emb.numpy())
        np.save(OUT_DIR / f"{slide_id}_patch_feats.npy", patch_feats.numpy())
        np.save(OUT_DIR / f"{slide_id}_coords.npy", patch_coords)

        rows_out.append({
            "slide_id": slide_id,
            "slide_path": str(slide_path),
            "target": target,
            "embedding_path": str(OUT_DIR / f"{slide_id}.npy"),
            "patch_feats_path": str(OUT_DIR / f"{slide_id}_patch_feats.npy"),
            "coords_path": str(OUT_DIR / f"{slide_id}_coords.npy"),
        })
        X.append(fused.numpy())
        y.append(target)

    X = np.stack(X, axis=0)
    y = np.asarray(y, dtype=np.float32)

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)

    pd.DataFrame(rows_out).to_csv(OUT_DIR / "manifest.csv", index=False)

    meta = {
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "patch_size_lv0": PATCH_SIZE_LV0,
        "embedding_dim": int(X.shape[1]),
        "n_slides": int(X.shape[0]),
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"Saved embeddings to: {OUT_DIR}")


if __name__ == "__main__":
    main()