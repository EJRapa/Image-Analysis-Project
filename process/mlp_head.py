import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_DIR = Path("/fast/rapae/BMED6460/group_project/tiger_titan_embeddings_no_pathology_report")

PATCH_DIR = FEATURE_DIR / "slide_model"
PATCH_DIR.mkdir(exist_ok=True)

MANIFEST_CSV = FEATURE_DIR / "manifest.csv"
X_SLIDE_PATH = FEATURE_DIR / "X.npy"
Y_PATH = FEATURE_DIR / "y.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE_SLIDES = 16


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

class SlideMLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden1=512, hidden2=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_slide_data():
    X = np.load(X_SLIDE_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.float32)
    df = pd.read_csv(MANIFEST_CSV).copy()

    if len(X) != len(y) or len(X) != len(df):
        raise RuntimeError(
            f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(df)={len(df)}"
        )

    return X, y, df


def train_slide_model(X, y, df):
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

    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42
    )

    X_train_np = X[train_idx]
    y_train_np = y[train_idx]
    X_val_np = X[val_idx]
    y_val_np = y[val_idx]

    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)
    X_val = torch.tensor(X_val_np, dtype=torch.float32, device=DEVICE)
    y_val = torch.tensor(y_val_np, dtype=torch.float32, device=DEVICE)

    model = SlideMLPRegressor(in_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    PATIENCE = 20
    MIN_DELTA = 1e-3

    best_state = None
    best_val_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()

        perm = torch.randperm(len(X_train), device=DEVICE)
        batch_losses = []

        for start in range(0, len(perm), BATCH_SIZE_SLIDES):
            idx = perm[start:start + BATCH_SIZE_SLIDES]

            xb = X_train[idx]
            yb = y_train[idx]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        model.eval()
        with torch.inference_mode():
            train_pred = model(X_train).detach().cpu().numpy()
            val_pred = model(X_val).detach().cpu().numpy()

            train_loss_eval = criterion(model(X_train), y_train).item()
            val_loss_eval = criterion(model(X_val), y_val).item()

        train_mae = mean_absolute_error(y_train_np, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_np, train_pred))
        train_r2 = r2_score(y_train_np, train_pred)

        val_mae = mean_absolute_error(y_val_np, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_np, val_pred))
        val_r2 = r2_score(y_val_np, val_pred)

        if val_mae < best_val_mae - MIN_DELTA:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_data["epoch"].append(epoch + 1)
        epoch_data["train_loss"].append(train_loss_eval)
        epoch_data["val_loss"].append(val_loss_eval)
        epoch_data["train_mae"].append(train_mae)
        epoch_data["val_mae"].append(val_mae)
        epoch_data["train_rmse"].append(train_rmse)
        epoch_data["val_rmse"].append(val_rmse)
        epoch_data["train_r2"].append(train_r2)
        epoch_data["val_r2"].append(val_r2)

        print(
            f"[slide] epoch={epoch+1:03d} "
            f"train_loss={train_loss_eval:.4f} "
            f"val_loss={val_loss_eval:.4f} "
            f"train_mae={train_mae:.4f} "
            f"val_mae={val_mae:.4f} "
            f"train_rmse={train_rmse:.4f} "
            f"val_rmse={val_rmse:.4f} "
            f"train_r2={train_r2:.4f} "
            f"val_r2={val_r2:.4f} "
            f"best_val_r2={best_val_mae:.4f} "
            f"patience={epochs_without_improvement}/{PATIENCE}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} with best val_r2={best_val_r2:.4f}")
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved during training.")

    model.load_state_dict(best_state)
    model.eval()

    epoch_df = pd.DataFrame.from_dict(epoch_data)
    epoch_df.to_csv(PATCH_DIR / "mlp_slide_results.csv", index=False)

    with torch.inference_mode():
        val_pred_raw = model(X_val).detach().cpu().numpy()

    # raw / fair metrics
    raw_metrics = regression_metrics(y_val_np, val_pred_raw)
    raw_metrics["model"] = "slide_mlp"

    print("\nValidation results (raw):")
    print(json.dumps(raw_metrics, indent=2))

    # post-hoc calibration on validation set
    calibrator = LinearRegression()
    calibrator.fit(val_pred_raw.reshape(-1, 1), y_val_np)
    val_pred_cal = calibrator.predict(val_pred_raw.reshape(-1, 1))

    cal_metrics = regression_metrics(y_val_np, val_pred_cal)
    cal_metrics["model"] = "slide_mlp_calibrated"
    cal_metrics["calibration_slope"] = float(calibrator.coef_[0])
    cal_metrics["calibration_intercept"] = float(calibrator.intercept_)

    print("\nValidation results (calibrated, optimistic):")
    print(json.dumps(cal_metrics, indent=2))

    val_out = df.iloc[val_idx].copy()
    val_out["pred_til_raw"] = val_pred_raw
    val_out["pred_til_calibrated"] = val_pred_cal
    val_out.to_csv(PATCH_DIR / "val_predictions.csv", index=False)

    train_out = df.iloc[train_idx].copy()
    train_out.to_csv(PATCH_DIR / "train_split_mlp.csv", index=False)
    val_out[["slide_id", "target", "pred_til_raw", "pred_til_calibrated"]].to_csv(
        PATCH_DIR / "val_split.csv",
        index=False
    )

    return model, val_pred_raw, val_pred_cal, raw_metrics, cal_metrics


def main():
    X, y, df = load_slide_data()

    model, val_pred_raw, val_pred_cal, raw_metrics, cal_metrics = train_slide_model(X, y, df)

    np.save(PATCH_DIR / "val_pred_raw.npy", val_pred_raw)
    np.save(PATCH_DIR / "val_pred_calibrated.npy", val_pred_cal)

    with open(PATCH_DIR / "val_results_raw.json", "w") as f:
        json.dump(raw_metrics, f, indent=2)

    with open(PATCH_DIR / "val_results_calibrated.json", "w") as f:
        json.dump(cal_metrics, f, indent=2)

    print(f"Saved results to: {PATCH_DIR}")


if __name__ == "__main__":
    main()