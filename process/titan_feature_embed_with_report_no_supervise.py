import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tiffslide import TiffSlide
from transformers import AutoModel


# =========================
# Paths / settings
# =========================

TCGA_WSI_ROOT = Path("/bulk/rapae/BMED6460/wsi_project/survival_data")

TCGA_WSI_DIR = TCGA_WSI_ROOT / "slides"
OUT_DIR = Path("../tcga_titan_embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 512          # model input patch size
STRIDE_20X = 512
PATCH_SIZE_20X_LV0 = 512

BATCH_SIZE = 8
MAX_PATCHES_PER_SLIDE = 15000
TISSUE_FRAC_THRESHOLD = 0.10
THUMB_MAX_SIDE = 4096

DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


# =========================
# Tissue mask from SVS
# =========================

def make_tissue_mask_from_slide(slide: TiffSlide, max_side: int = THUMB_MAX_SIDE) -> np.ndarray:
    w, h = slide.dimensions

    scale = max(w, h) / max_side
    if scale < 1:
        scale = 1.0

    out_w = max(1, int(w / scale))
    out_h = max(1, int(h / scale))

    thumb = slide.get_thumbnail((out_w, out_h)).convert("RGB")
    thumb_np = np.array(thumb)

    gray = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return (mask > 0).astype(np.uint8)


# =========================
# Patch coordinate generation
# =========================

def get_objective_power(slide: TiffSlide) -> float:
    val = slide.properties.get("openslide.objective-power", None)

    if val is None:
        print("Warning: missing objective power; assuming 20x")
        return 20.0

    try:
        return float(val)
    except ValueError:
        print(f"Warning: bad objective power {val}; assuming 20x")
        return 20.0


def get_level0_patch_params(slide):
    obj = slide.properties.get("openslide.objective-power", None)

    if obj is not None:
        objective = float(obj)
    else:
        mpp = slide.properties.get("openslide.mpp-x", None)
        if mpp is not None:
            mpp = float(mpp)
            objective = 40.0 if mpp < 0.35 else 20.0
            print(f"Inferred objective {objective}x from mpp-x={mpp}", flush=True)
        else:
            objective = 20.0
            print("Warning: missing objective and mpp; assuming 20x", flush=True)

    if objective >= 40:
        return 1024, 1024, objective
    else:
        return 512, 512, objective


def generate_level0_coords(
    slide: TiffSlide,
    patch_size_lv0: int,
    stride_lv0: int,
    tissue_mask: np.ndarray,
    tissue_threshold: float = TISSUE_FRAC_THRESHOLD,
) -> np.ndarray:

    slide_w, slide_h = slide.dimensions
    mask_h, mask_w = tissue_mask.shape

    xs = np.arange(
        0,
        max(1, slide_w - patch_size_lv0 + 1),
        stride_lv0,
        dtype=np.int64,
    )

    ys = np.arange(
        0,
        max(1, slide_h - patch_size_lv0 + 1),
        stride_lv0,
        dtype=np.int64,
    )

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

    coords = np.asarray(coords, dtype=np.int64)

    if len(coords) > MAX_PATCHES_PER_SLIDE:
        print(f"Subsampling {len(coords)} patches to {MAX_PATCHES_PER_SLIDE}", flush=True)
        rng = np.random.default_rng(42)
        keep_idx = rng.choice(len(coords), size=MAX_PATCHES_PER_SLIDE, replace=False)
        keep_idx = np.sort(keep_idx)
        coords = coords[keep_idx]

    return np.asarray(coords, dtype=np.int64)


# =========================
# Patch reading / encoding
# =========================

def read_patches(
    slide: TiffSlide,
    coords: np.ndarray,
    patch_size_lv0: int,
) -> list:

    patches = []

    for x, y in coords:
        patch = slide.read_region(
            (int(x), int(y)),
            0,
            (patch_size_lv0, patch_size_lv0),
        ).convert("RGB")

        if patch_size_lv0 != PATCH_SIZE:
            patch = patch.resize((PATCH_SIZE, PATCH_SIZE))

        patches.append(patch)

    return patches


def conch_encode_batch(conch, images_tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = conch(images_tensor)

        if isinstance(out, (tuple, list)):
            feats = out[0]
        elif isinstance(out, dict):
            feats = out.get("image_features", None)
            if feats is None:
                raise RuntimeError(f"Could not find image_features in CONCH output keys: {out.keys()}")
        else:
            feats = out

    return feats


def extract_slide_embedding(
    slide_path: Path,
    titan,
    conch,
    eval_transform,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:

    slide = TiffSlide(str(slide_path))

    patch_size_lv0, stride_lv0, objective = get_level0_patch_params(slide)

    tissue_mask = make_tissue_mask_from_slide(slide)

    coords = generate_level0_coords(
        slide=slide,
        patch_size_lv0=patch_size_lv0,
        stride_lv0=stride_lv0,
        tissue_mask=tissue_mask,
        tissue_threshold=TISSUE_FRAC_THRESHOLD,
    )

    if len(coords) == 0:
        raise RuntimeError(f"No tissue patches found for {slide_path.name}")

    all_features = []

    print(f"{slide_path.name}: {len(coords)} tissue patches", flush=True)

    for start in range(0, len(coords), BATCH_SIZE):
        batch_coords = coords[start:start + BATCH_SIZE]
        batch_pil = read_patches(slide, batch_coords, patch_size_lv0)

        batch_tensor = torch.stack(
            [eval_transform(im) for im in batch_pil],
            dim=0,
        ).to(DEVICE)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda"),
        ):
            batch_feats = conch_encode_batch(conch, batch_tensor)

        all_features.append(batch_feats.detach().float().cpu())

        del batch_tensor, batch_feats
        torch.cuda.empty_cache()

    features_cpu = torch.cat(all_features, dim=0)

    features = features_cpu.to(DEVICE)
    coords_t = torch.from_numpy(coords).to(DEVICE)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(DEVICE.type == "cuda"),
    ), torch.inference_mode():

        slide_embedding = titan.encode_slide_from_patch_features(
            features,
            coords_t,
            patch_size_lv0,
        )

    slide_meta = {
        "objective_power": objective,
        "patch_size_model": PATCH_SIZE,
        "patch_size_lv0": patch_size_lv0,
        "stride_lv0": stride_lv0,
        "n_patches": int(len(coords)),
        "slide_width": int(slide.dimensions[0]),
        "slide_height": int(slide.dimensions[1]),
    }

    slide_embedding_cpu = slide_embedding.detach().float().cpu().squeeze()

    del features
    del coords_t
    del slide_embedding
    torch.cuda.empty_cache()

    return (
        slide_embedding_cpu,
        features_cpu,
        coords,
        slide_meta,
    )


# =========================
# Optional text encoding
# =========================

def encode_notes(notes: str, titan) -> torch.Tensor:
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


# =========================
# Main
# =========================

def main():
    print("Script started", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    print(f"Looking for slides in: {TCGA_WSI_DIR}", flush=True)

    titan = AutoModel.from_pretrained(
        "MahmoodLab/TITAN",
        trust_remote_code=True,
    ).to(DEVICE).eval()

    conch, eval_transform = titan.return_conch()
    conch = conch.to(DEVICE).eval()

    slide_paths = sorted(TCGA_WSI_DIR.rglob("*.svs"))

    if len(slide_paths) == 0:
        raise RuntimeError(f"No .svs files found under {TCGA_WSI_DIR}")

    rows_out = []
    X = []

    for slide_path in slide_paths:
        slide_id = slide_path.stem

        print(f"\nProcessing {slide_path.name}")

        try:
            emb, patch_feats, patch_coords, slide_meta = extract_slide_embedding(
                slide_path=slide_path,
                titan=titan,
                conch=conch,
                eval_transform=eval_transform,
            )

        except Exception as e:
            print(f"Failed on {slide_path.name}: {e}")
            torch.cuda.empty_cache()
            rows_out.append({
                "slide_id": slide_id,
                "slide_path": str(slide_path),
                "status": "failed",
                "error": str(e),
            })
            continue

        emb_path = OUT_DIR / f"{slide_id}.npy"
        patch_feats_path = OUT_DIR / f"{slide_id}_patch_feats.npy"
        coords_path = OUT_DIR / f"{slide_id}_coords.npy"
        slide_meta_path = OUT_DIR / f"{slide_id}_meta.json"

        emb_np = emb.numpy()

        np.save(emb_path, emb_np)
        np.save(patch_feats_path, patch_feats.numpy())
        np.save(coords_path, patch_coords)

        with open(slide_meta_path, "w") as f:
            json.dump(slide_meta, f, indent=2)

        rows_out.append({
            "slide_id": slide_id,
            "slide_path": str(slide_path),
            "status": "ok",
            "embedding_path": str(emb_path),
            "patch_feats_path": str(patch_feats_path),
            "coords_path": str(coords_path),
            "slide_meta_path": str(slide_meta_path),
            **slide_meta,
        })

        X.append(emb.numpy())

        del emb, emb_np, patch_feats, patch_coords, slide_meta
        torch.cuda.empty_cache()

    manifest = pd.DataFrame(rows_out)
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)

    if len(X) > 0:
        X = np.stack(X, axis=0)
        np.save(OUT_DIR / "X.npy", X)

        dataset_meta = {
            "n_successful_slides": int(X.shape[0]),
            "embedding_dim": int(X.shape[1]),
            "patch_size_model": PATCH_SIZE,
            "tissue_frac_threshold": TISSUE_FRAC_THRESHOLD,
            "note": "TCGA/GDC SVS slides processed without external masks; tissue masks generated from slide thumbnails.",
        }

        with open(OUT_DIR / "metadata.json", "w") as f:
            json.dump(dataset_meta, f, indent=2)

    print("\nDone.", flush=True)
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()