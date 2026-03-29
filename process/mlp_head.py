import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_DIR = Path("./tiger_uni2h_features")
MANIFEST_CSV = FEATURE_DIR / "manifest.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS = 5
EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4
FEAT_DIM = 1536


def load_bag(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:].astype(np.float32)
        coords = f["coords"][:].astype(np.int64)
    return feats, coords


class MeanPoolRegressor(nn.Module):
    def __init__(self, in_dim=1536, hidden=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        pooled = x.mean(dim=0, keepdim=True)
        return self.head(pooled).squeeze(0).squeeze(0)


class AttentionMILRegressor(nn.Module):
    def __init__(self, in_dim=1536, attn_dim=256, hidden=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        a = self.attn(x)                    # [N, 1]
        w = torch.softmax(a.squeeze(1), dim=0)
        pooled = torch.sum(w[:, None] * x, dim=0, keepdim=True)
        return self.head(pooled).squeeze(0).squeeze(0)


def train_one_fold(train_df, val_df, model_type="attn"):
    if model_type == "mean":
        model = MeanPoolRegressor(in_dim=FEAT_DIM).to(DEVICE)
    else:
        model = AttentionMILRegressor(in_dim=FEAT_DIM).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_state = None
    best_val_mae = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        train_df_shuffled = train_df.sample(frac=1.0, random_state=epoch)
        for _, row in train_df_shuffled.iterrows():
            feats, _ = load_bag(row["feature_h5"])
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            y = torch.tensor(float(row["target"]), dtype=torch.float32, device=DEVICE)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for _, row in val_df.iterrows():
                feats, _ = load_bag(row["feature_h5"])
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                pred = model(x).item()
                y_true.append(float(row["target"]))
                y_pred.append(pred)

        val_mae = mean_absolute_error(y_true, y_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch={epoch+1:03d} "
            f"train_loss={np.mean(train_losses):.4f} "
            f"val_mae={val_mae:.4f}"
        )

    model.load_state_dict(best_state)
    model.eval()

    final_true, final_pred = [], []
    with torch.inference_mode():
        for _, row in val_df.iterrows():
            feats, _ = load_bag(row["feature_h5"])
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
            pred = model(x).item()
            final_true.append(float(row["target"]))
            final_pred.append(pred)

    return np.array(final_true), np.array(final_pred)


def main():
    df = pd.read_csv(MANIFEST_CSV)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for model_type in ["mean", "attn"]:
        print(f"\n=== Training model_type={model_type} ===")

        oof_pred = np.zeros(len(df), dtype=np.float32)
        metrics = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(df), start=1):
            print(f"\n--- Fold {fold} ---")
            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df = df.iloc[va_idx].reset_index(drop=True)

            y_true, y_pred = train_one_fold(train_df, val_df, model_type=model_type)
            oof_pred[va_idx] = y_pred

            fold_mae = mean_absolute_error(y_true, y_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            fold_r2 = r2_score(y_true, y_pred)

            metrics.append({
                "fold": fold,
                "mae": float(fold_mae),
                "rmse": float(fold_rmse),
                "r2": float(fold_r2),
            })

            print(f"Fold {fold}: MAE={fold_mae:.4f} RMSE={fold_rmse:.4f} R2={fold_r2:.4f}")

        overall = {
            "model_type": model_type,
            "cv_mae": float(mean_absolute_error(df["target"].values, oof_pred)),
            "cv_rmse": float(np.sqrt(mean_squared_error(df["target"].values, oof_pred))),
            "cv_r2": float(r2_score(df["target"].values, oof_pred)),
            "folds": metrics,
        }

        print("\nOverall:")
        print(json.dumps(overall, indent=2))

        np.save(FEATURE_DIR / f"oof_pred_{model_type}.npy", oof_pred)
        with open(FEATURE_DIR / f"cv_results_{model_type}.json", "w") as f:
            json.dump(overall, f, indent=2)


if __name__ == "__main__":
    main()