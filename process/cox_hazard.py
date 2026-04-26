import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedKFold

TILSCOUT_TILS_FOLDER = Path("/fast/rapae/BMED6460/group_project/TILScout/results")
TITAN_TILS_FOLDER = Path("/fast/rapae/BMED6460/group_project/tiger_titan_embeddings_fused_text/patch_model_full_train")

TITAN_TILS_FILE = TITAN_TILS_FOLDER / "tcga_predicted_til_scores.csv"

OUT_DIR = Path("/fast/rapae/BMED6460/group_project/tiger_titan_embeddings_fused_text/patch_model_full_train/cox")

OUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_JSON = Path("/bulk/rapae/BMED6460/wsi_project/survival_data/clinical.cart.2026-04-26.json")

WSI_NAME_TTL = "WSI_NAME"
TILS_SCORE_TTL = "predicted_til_score"

N_SPLITS = 5
SEED = 42
PENALIZER = 0.1

def get_TILScout_TILS(results_folder: Path):

    rows = []

    for i, file in enumerate(results_folder.glob("*.txt")):
        
        text = file.read_text().strip()
        title, til_score = text.split()
        
        rows.append({
            WSI_NAME_TTL: title,
            TILS_SCORE_TTL: float(til_score)
        })

    df = pd.DataFrame(rows)

    return df

def get_titan_TILS(results_folder: Path): 

    rows = []

    with open(results_folder) as f:
        next(f)

        for line in f:
            slide_id, _, predicted_til_score = line.split(sep=',')

            rows.append({
                    WSI_NAME_TTL: slide_id,
                    TILS_SCORE_TTL: float(predicted_til_score)
                })

    df = pd.DataFrame(rows)        


    return df


def slide_to_patient_id(slide_id: str) -> str:
    base = Path(str(slide_id)).stem
    return "-".join(base.split("-")[:3])


def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def max_valid(values):
    values = [safe_float(v) for v in values]
    values = [v for v in values if not np.isnan(v)]

    if len(values) == 0:
        return np.nan

    return max(values)

def extract_survival_from_case(case: dict) -> dict:
    patient_id = case.get("submitter_id")

    demographic = case.get("demographic", {}) or {}
    vital_status = str(demographic.get("vital_status", "")).strip().lower()

    event = 1 if vital_status == "dead" else 0

    diagnoses = case.get("diagnoses", []) or []
    follow_ups = case.get("follow_ups", []) or []

    days_to_death_vals = []
    days_to_last_follow_up_vals = []
    days_to_follow_up_vals = []

    for dx in diagnoses:
        days_to_death_vals.append(dx.get("days_to_death"))
        days_to_last_follow_up_vals.append(dx.get("days_to_last_follow_up"))

    for fu in follow_ups:
        days_to_death_vals.append(fu.get("days_to_death"))
        days_to_follow_up_vals.append(fu.get("days_to_follow_up"))
        days_to_last_follow_up_vals.append(fu.get("days_to_last_follow_up"))

    days_to_death = max_valid(days_to_death_vals)
    days_to_last_follow_up = max_valid(days_to_last_follow_up_vals + days_to_follow_up_vals)

    if event == 1:
        duration = days_to_death
        if np.isnan(duration):
            duration = days_to_last_follow_up
    else:
        duration = days_to_last_follow_up

    return {
        "patient_id": patient_id,
        "vital_status": vital_status,
        "duration": duration,
        "event": event,
    }


def load_clinical_survival(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r") as f:
        cases = json.load(f)

    rows = [extract_survival_from_case(case) for case in cases]
    df = pd.DataFrame(rows)

    df = df.dropna(subset=["patient_id", "duration", "event"])
    df = df[df["duration"] > 0].copy()

    return df

def main():
    clinical_df = load_clinical_survival(CLINICAL_JSON)

    # pred_df = get_TILScout_TILS(TILSCOUT_TILS_FOLDER)
    pred_df = get_titan_TILS(TITAN_TILS_FILE)

    pred_df["patient_id"] = pred_df[WSI_NAME_TTL].apply(slide_to_patient_id)

    patient_pred = (
        pred_df
        .groupby("patient_id", as_index=False)
        .agg(predicted_til_score=(TILS_SCORE_TTL, "mean"),
             n_slides=(WSI_NAME_TTL, "count"))
    )

    df = patient_pred.merge(clinical_df, on="patient_id", how="inner")

    df = df.dropna(subset=[TILS_SCORE_TTL, "duration", "event"]).copy()

    print("Merged survival dataset:")
    print(df[["patient_id", TILS_SCORE_TTL, "duration", "event", "n_slides"]].head())
    print(f"N patients: {len(df)}")
    print(f"Events: {df['event'].sum()}")

    df.to_csv(OUT_DIR / "cox_input_table.csv", index=False)

    model_df = df[["patient_id", TILS_SCORE_TTL, "duration", "event"]].copy()

    oof_rows = []
    fold_rows = []

    n_events = int(model_df["event"].sum())
    n_nonevents = int((model_df["event"] == 0).sum())
    n_splits = min(N_SPLITS, n_events, n_nonevents)

    if n_splits < 2:
        raise RuntimeError(f"Not enough events/non-events for CV: events={n_events}, non-events={n_nonevents}")

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(
        kf.split(model_df, model_df["event"]),
        start=1
    ):
        train = model_df.iloc[train_idx].copy()
        test = model_df.iloc[test_idx].copy()

        cph = CoxPHFitter(penalizer=PENALIZER)

        cph.fit(
            train[[TILS_SCORE_TTL, "duration", "event"]],
            duration_col="duration",
            event_col="event",
        )

        test_risk = cph.predict_partial_hazard(
            test[[TILS_SCORE_TTL]]
        ).values.ravel()

        cidx = concordance_index(
            test["duration"],
            -test_risk,
            test["event"],
        )

        fold_rows.append({
            "fold": fold,
            "c_index": cidx,
            "coef_til": cph.params_[TILS_SCORE_TTL],
            "hr_til": np.exp(cph.params_[TILS_SCORE_TTL]),
            "n_train": len(train),
            "n_test": len(test),
            "events_test": int(test["event"].sum()),
        })

        fold_oof = test[["patient_id", TILS_SCORE_TTL, "duration", "event"]].copy()
        fold_oof["fold"] = fold
        fold_oof["risk"] = test_risk
        oof_rows.append(fold_oof)

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_rows, axis=0).reset_index(drop=True)

    pooled_cindex = concordance_index(
        oof_df["duration"],
        -oof_df["risk"],
        oof_df["event"],
    )

    print("\nFold results:")
    print(fold_df)
    print(f"\nMean CV C-index: {fold_df['c_index'].mean():.3f} ± {fold_df['c_index'].std():.3f}")
    print(f"Pooled OOF C-index: {pooled_cindex:.3f}")

    fold_df.to_csv(OUT_DIR / "cv_fold_results.csv", index=False)
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    final_cph = CoxPHFitter(penalizer=PENALIZER)
    final_cph.fit(
        model_df[[TILS_SCORE_TTL, "duration", "event"]],
        duration_col="duration",
        event_col="event",
    )

    final_summary = final_cph.summary
    final_summary.to_csv(OUT_DIR / "final_cox_summary.csv")

    print("\nFinal Cox summary:")
    print(final_summary)

    median_risk = oof_df["risk"].median()
    oof_df["risk_group"] = np.where(oof_df["risk"] >= median_risk, "High predicted risk", "Low predicted risk")

    high = oof_df[oof_df["risk_group"] == "High predicted risk"]
    low = oof_df[oof_df["risk_group"] == "Low predicted risk"]

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(7, 5))

    kmf.fit(
        durations=low["duration"],
        event_observed=low["event"],
        label=f"Low risk (n={len(low)})",
    )
    ax = kmf.plot_survival_function(ci_show=True)

    kmf.fit(
        durations=high["duration"],
        event_observed=high["event"],
        label=f"High risk (n={len(high)})",
    )
    kmf.plot_survival_function(ax=ax, ci_show=True)

    result = logrank_test(
        low["duration"],
        high["duration"],
        event_observed_A=low["event"],
        event_observed_B=high["event"],
        SEED=SEED
    )

    plt.title(f"KM curve by CV-predicted risk\nLog-rank p = {result.p_value:.3g}")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "km_curve_oof_risk_groups.png", dpi=300)
    plt.savefig(OUT_DIR / "km_curve_oof_risk_groups.pdf")
    plt.close()

    oof_df.to_csv(OUT_DIR / "oof_predictions_with_risk_groups.csv", index=False)

    print(f"\nSaved results to: {OUT_DIR}")


if __name__ == "__main__":
    main()