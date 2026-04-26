import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

FEATURE_DIR = Path("/fast/rapae/BMED6460/group_project/tiger_titan_embeddings_fused_text")

PATCH_DIR = FEATURE_DIR / "patch_model_full_train"
PATCH_DIR.mkdir(exist_ok=True)

MANIFEST_CSV = FEATURE_DIR / "manifest.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 190
LR = 1e-5
WEIGHT_DECAY = 1e-5

def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

def load_patch_bag(patch_feats_path):
    feats = np.load(patch_feats_path).astype(np.float32)
    return feats

class PatchTILWeightedRatioRegressor(nn.Module):
    def __init__(self, in_dim, hidden1=256, hidden2=64, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.p_head = nn.Linear(hidden2, 1)
        self.w_head = nn.Linear(hidden2, 1)

    def forward(self, patch_feats):
        # patch_feats: [N, D]
        h = self.backbone(patch_feats)                           # [N, hidden2]
        p = torch.sigmoid(self.p_head(h)).squeeze(1)            # [N]
        w = torch.nn.functional.softplus(self.w_head(h)).squeeze(1)  # [N]

        slide_pred = (w * p).sum() / (w.sum() + 1e-8)          # scalar in [0,1]
        return slide_pred, p, w


def train_patch_weighted_ratio(df):
    epoch_data = {
        "epoch": [],
        "train_loss": [],
        "train_mae": [],
        "train_rmse": [],
        "train_r2": [],
    }

    feat_dim = load_patch_bag(df.iloc[0]["patch_feats_path"]).shape[1]
    print(f"Using patch feature dimension: {feat_dim}", flush=True)

    model = PatchTILWeightedRatioRegressor(in_dim=feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    scale = 100

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        df_shuffled = df.sample(frac=1.0, random_state=epoch)

        for _, row in df_shuffled.iterrows():
            feats = load_patch_bag(row["patch_feats_path"])
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

            y_true_scaled = torch.tensor(
                float(row["target"]) / scale,
                dtype=torch.float32,
                device=DEVICE,
            )

            slide_pred, p, w = model(x)
            loss = criterion(slide_pred, y_true_scaled)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            del x, y_true_scaled, slide_pred, p, w, loss
            torch.cuda.empty_cache()

        model.eval()

        train_true_list = []
        train_pred_list = []
        train_eval_losses = []

        with torch.inference_mode():
            for _, row in df.iterrows():
                feats = load_patch_bag(row["patch_feats_path"])
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

                y_true = float(row["target"])
                y_true_scaled = torch.tensor(
                    y_true / scale,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                slide_pred, p, w = model(x)
                loss = criterion(slide_pred, y_true_scaled)

                train_eval_losses.append(loss.item())
                train_true_list.append(y_true)
                train_pred_list.append(float(slide_pred.item() * scale))

                del x, y_true_scaled, slide_pred, p, w, loss
                torch.cuda.empty_cache()

        train_loss_mean = float(np.mean(train_eval_losses))
        train_mae = mean_absolute_error(train_true_list, train_pred_list)
        train_rmse = np.sqrt(mean_squared_error(train_true_list, train_pred_list))
        train_r2 = r2_score(train_true_list, train_pred_list)

        epoch_data["epoch"].append(epoch + 1)
        epoch_data["train_loss"].append(train_loss_mean)
        epoch_data["train_mae"].append(train_mae)
        epoch_data["train_rmse"].append(train_rmse)
        epoch_data["train_r2"].append(train_r2)

        print(
            f"[full_train] epoch={epoch+1:03d} "
            f"loss={train_loss_mean:.4f} "
            f"mae={train_mae:.4f} "
            f"rmse={train_rmse:.4f} "
            f"r2={train_r2:.4f}",
            flush=True,
        )

    pd.DataFrame(epoch_data).to_csv(PATCH_DIR / "full_train_results.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feat_dim": feat_dim,
            "target_scale": scale,
        },
        PATCH_DIR / "patch_weighted_ratio_full_train.pt",
    )

    return model, scale, feat_dim

def predict_patch_outputs(model, patch_feats_path, target_scale=1.0):
    model.eval()
    feats = load_patch_bag(patch_feats_path)
    x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        slide_pred, p, w = model(x)

    return {
        "slide_pred": float(slide_pred.item() * target_scale),
        "patch_til_prob": p.detach().cpu().numpy(),
        "patch_weight": w.detach().cpu().numpy(),
    }

def predict_tcga_til_scores(model, tcga_manifest_csv, out_csv, target_scale):
    tcga_df = pd.read_csv(tcga_manifest_csv).copy()

    rows = []

    for _, row in tcga_df.iterrows():
        if "status" in row and row["status"] != "ok":
            continue

        patch_feats_path = row["patch_feats_path"]

        out = predict_patch_outputs(
            model,
            patch_feats_path,
            target_scale=target_scale,
        )

        rows.append({
            "slide_id": row["slide_id"],
            "slide_path": row.get("slide_path", ""),
            "predicted_til_score": out["slide_pred"],
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_csv, index=False)

    print(f"Saved TCGA predicted TIL scores to: {out_csv}", flush=True)

    return pred_df

def main():
    wsitils_df = pd.read_csv(MANIFEST_CSV).copy()

    model, scale, feat_dim = train_patch_weighted_ratio(wsitils_df)

    TCGA_MANIFEST_CSV = Path("../tcga_titan_embeddings/manifest.csv")
    TCGA_TIL_PRED_CSV = PATCH_DIR / "tcga_predicted_til_scores.csv"

    predict_tcga_til_scores(
        model=model,
        tcga_manifest_csv=TCGA_MANIFEST_CSV,
        out_csv=TCGA_TIL_PRED_CSV,
        target_scale=scale,
    )

    meta = {
        "training_data": "WSI-TILs full training set",
        "test_data": "TCGA external set",
        "feat_dim": int(feat_dim),
        "target_scale": float(scale),
        "tcga_prediction_csv": str(TCGA_TIL_PRED_CSV),
    }

    with open(PATCH_DIR / "full_train_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()