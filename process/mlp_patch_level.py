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

PATCH_DIR = FEATURE_DIR / "patch_model"
PATCH_DIR.mkdir(exist_ok=True)

MANIFEST_CSV = FEATURE_DIR / "manifest.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 200
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


def train_patch_weighted_ratio_model(df):
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    epoch_data = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_r2": [],
        "val_r2": [],
    }

    feat_dim = load_patch_bag(df.iloc[0]["patch_feats_path"]).shape[1]
    print(f"Using patch feature dimension: {feat_dim}")

    model = PatchTILWeightedRatioRegressor(in_dim=feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    PATIENCE = 20
    MIN_DELTA = 1e-4
    scale = 100

    best_state = None
    best_val_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        train_df_shuffled = train_df.sample(frac=1.0, random_state=epoch)

        for _, row in train_df_shuffled.iterrows():
            feats = load_patch_bag(row["patch_feats_path"])
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

            y_true_scaled = float(row["target"]) / scale
            y_true_scaled = torch.tensor(y_true_scaled, dtype=torch.float32, device=DEVICE)

            slide_pred, p, w = model(x)
            loss = criterion(slide_pred, y_true_scaled)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()

        train_true_list, train_pred_list, train_eval_losses = [], [], []
        val_true_list, val_pred_list, val_losses = [], [], []

        with torch.inference_mode():
            for _, row in train_df.iterrows():
                feats = load_patch_bag(row["patch_feats_path"])
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

                y_true = float(row["target"])
                y_true_scaled = torch.tensor(y_true / scale, dtype=torch.float32, device=DEVICE)

                slide_pred, p, w = model(x)
                loss = criterion(slide_pred, y_true_scaled)

                train_eval_losses.append(loss.item())
                train_true_list.append(y_true)
                train_pred_list.append(float(slide_pred.item() * scale))

            for _, row in val_df.iterrows():
                feats = load_patch_bag(row["patch_feats_path"])
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

                y_true = float(row["target"])
                y_true_scaled = torch.tensor(y_true / scale, dtype=torch.float32, device=DEVICE)

                slide_pred, p, w = model(x)
                loss = criterion(slide_pred, y_true_scaled)

                val_losses.append(loss.item())
                val_true_list.append(y_true)
                val_pred_list.append(float(slide_pred.item() * scale))

        train_loss_mean = float(np.mean(train_eval_losses))
        val_loss_mean = float(np.mean(val_losses))

        train_mae = mean_absolute_error(train_true_list, train_pred_list)
        train_rmse = np.sqrt(mean_squared_error(train_true_list, train_pred_list))
        train_r2 = r2_score(train_true_list, train_pred_list)

        val_mae = mean_absolute_error(val_true_list, val_pred_list)
        val_rmse = np.sqrt(mean_squared_error(val_true_list, val_pred_list))
        val_r2 = r2_score(val_true_list, val_pred_list)

        if val_mae < best_val_mae - MIN_DELTA:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_data["epoch"].append(epoch + 1)
        epoch_data["train_loss"].append(train_loss_mean)
        epoch_data["val_loss"].append(val_loss_mean)
        epoch_data["train_mae"].append(train_mae)
        epoch_data["val_mae"].append(val_mae)
        epoch_data["train_rmse"].append(train_rmse)
        epoch_data["val_rmse"].append(val_rmse)
        epoch_data["train_r2"].append(train_r2)
        epoch_data["val_r2"].append(val_r2)

        print(
            f"[patch_ratio] epoch={epoch+1:03d} "
            f"train_loss={train_loss_mean:.4f} "
            f"val_loss={val_loss_mean:.4f} "
            f"train_mae={train_mae:.4f} "
            f"val_mae={val_mae:.4f} "
            f"train_r2={train_r2:.4f} "
            f"val_r2={val_r2:.4f} "
            f"best_val_mae={best_val_mae:.4f} "
            f"patience={epochs_without_improvement}/{PATIENCE}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} with best val_mae={best_val_mae:.4f}")
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved during training.")

    model.load_state_dict(best_state)
    model.eval()

    df_epoch_data = pd.DataFrame.from_dict(epoch_data)
    df_epoch_data.to_csv(PATCH_DIR / "mlp_results.csv", index=False)

    final_true, final_pred = [], []

    with torch.inference_mode():
        for _, row in val_df.iterrows():
            feats = load_patch_bag(row["patch_feats_path"])
            x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

            slide_pred, p, w = model(x)

            final_true.append(float(row["target"]))
            final_pred.append(float(slide_pred.item() * scale))

    final_true = np.array(final_true, dtype=np.float32)
    final_pred = np.array(final_pred, dtype=np.float32)

    # Raw / fair metrics
    raw_metrics = regression_metrics(final_true, final_pred)
    raw_metrics["model"] = "patch_weighted_ratio"
    raw_metrics["target_scale"] = scale
    raw_metrics["patch_feat_dim"] = feat_dim

    print("\nValidation results (raw):")
    print(json.dumps(raw_metrics, indent=2))

    # Post-hoc calibration on validation set
    calibrator = LinearRegression()
    calibrator.fit(final_pred.reshape(-1, 1), final_true)

    final_pred_cal = calibrator.predict(final_pred.reshape(-1, 1))

    cal_metrics = regression_metrics(final_true, final_pred_cal)
    cal_metrics["model"] = "patch_weighted_ratio_calibrated"
    cal_metrics["target_scale"] = scale
    cal_metrics["patch_feat_dim"] = feat_dim
    cal_metrics["calibration_slope"] = float(calibrator.coef_[0])
    cal_metrics["calibration_intercept"] = float(calibrator.intercept_)

    print("\nValidation results (calibrated, optimistic):")
    print(json.dumps(cal_metrics, indent=2))

    return model, final_pred, final_pred_cal, raw_metrics, cal_metrics, train_df, val_df, scale

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

def main():
    df = pd.read_csv(MANIFEST_CSV).copy()

    model, val_pred_raw, val_pred_cal, raw_metrics, cal_metrics, train_df, val_df, scale = train_patch_weighted_ratio_model(df)

    np.save(PATCH_DIR / "val_pred_raw.npy", val_pred_raw)
    np.save(PATCH_DIR / "val_pred_calibrated.npy", val_pred_cal)

    with open(PATCH_DIR / "val_results_raw.json", "w") as f:
        json.dump(raw_metrics, f, indent=2)

    with open(PATCH_DIR / "val_results_calibrated.json", "w") as f:
        json.dump(cal_metrics, f, indent=2)

    val_out = val_df.copy()
    val_out["pred_til_raw"] = val_pred_raw
    val_out["pred_til_calibrated"] = val_pred_cal
    val_out.to_csv(PATCH_DIR / "val_predictions.csv", index=False)

    train_df.to_csv(PATCH_DIR / "train_split_mlp.csv", index=False)
    val_df.to_csv(PATCH_DIR / "val_split.csv", index=False)

    print(f"Saved results to: {PATCH_DIR}")


if __name__ == "__main__":
    main()