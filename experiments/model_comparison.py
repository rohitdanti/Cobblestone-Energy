"""
Model comparison experiment — NOT part of the pipeline.

Loads the existing feature matrix and splits, runs 7 candidate models,
and prints a ranked comparison table.

Run:
    ./.venv/bin/python experiments/model_comparison.py
"""
import sys
import time
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import compute_metrics, seasonal_naive_predict
from src.models.extra_trees_model import FEATURE_COLS

# ── Load data ─────────────────────────────────────────────────────────────────

cfg = yaml.safe_load((PROJECT_ROOT / "config/config.yaml").read_text())
sc = cfg["splits"]

df = pd.read_parquet(PROJECT_ROOT / "processed/features.parquet")
df = df.dropna(subset=FEATURE_COLS + ["target"])

train_end = pd.Timestamp(f"{sc['train_end']} 23:00", tz="UTC")
val_end   = pd.Timestamp(f"{sc['validation_end']} 23:00", tz="UTC")
val_start = train_end + pd.Timedelta(hours=1)
hold_start = val_end + pd.Timedelta(hours=1)

train_df = df[df.index <= train_end]
val_df   = df[(df.index >= val_start) & (df.index <= val_end)]
hold_df  = df[df.index >= hold_start]

X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
X_val,   y_val   = val_df[FEATURE_COLS],   val_df["target"]
X_hold,  y_hold  = hold_df[FEATURE_COLS],  hold_df["target"]

print(f"Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | Holdout: {len(hold_df):,} rows\n")

# ── Candidate models ──────────────────────────────────────────────────────────

def _make_candidates():
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import lightgbm as lgb
    import xgboost as xgb

    m = cfg["model"]

    return {
        "01_seasonal_naive": None,  # handled separately
        "02_ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "03_lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=1.0, max_iter=5000)),
        ]),
        "04_elasticnet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)),
        ]),
        "05_random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            n_jobs=-1, random_state=42,
        ),
        "06_extra_trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            n_jobs=-1, random_state=42,
        ),
        "07_gradient_boosting_sklearn": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42,
        ),
        "08_xgboost": xgb.XGBRegressor(
            n_estimators=m["n_estimators"],
            learning_rate=m["learning_rate"],
            max_depth=m["max_depth"],
            subsample=m["subsample"],
            colsample_bytree=m["colsample_bytree"],
            random_state=m["random_state"],
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50,
        ),
        "09_lightgbm": lgb.LGBMRegressor(
            n_estimators=m["n_estimators"],
            learning_rate=m["learning_rate"],
            max_depth=m["max_depth"],
            num_leaves=m["num_leaves"],
            min_child_samples=m["min_child_samples"],
            subsample=m["subsample"],
            colsample_bytree=m["colsample_bytree"],
            random_state=m["random_state"],
            n_jobs=-1,
            verbosity=-1,
        ),
    }

# ── Run comparison ────────────────────────────────────────────────────────────

results = []
candidates = _make_candidates()

for name, model in candidates.items():
    t0 = time.time()

    if name == "01_seasonal_naive":
        val_preds  = seasonal_naive_predict(val_df)
        hold_preds = seasonal_naive_predict(hold_df)
    elif name == "08_xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_preds  = pd.Series(model.predict(X_val),  index=val_df.index)
        hold_preds = pd.Series(model.predict(X_hold), index=hold_df.index)
    elif name == "09_lightgbm":
        import lightgbm as lgb
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_preds  = pd.Series(model.predict(X_val),  index=val_df.index)
        hold_preds = pd.Series(model.predict(X_hold), index=hold_df.index)
    else:
        model.fit(X_train, y_train)
        val_preds  = pd.Series(model.predict(X_val),  index=val_df.index)
        hold_preds = pd.Series(model.predict(X_hold), index=hold_df.index)

    elapsed = time.time() - t0

    vm = compute_metrics(val_df["target"],  val_preds,  label=name)
    hm = compute_metrics(hold_df["target"], hold_preds, label=name)

    results.append({
        "model":       name,
        "val_mae":     vm["mae"],
        "val_rmse":    vm["rmse"],
        "hold_mae":    hm["mae"],
        "hold_rmse":   hm["rmse"],
        "tail_hi":     hm["tail_mae_high"],
        "tail_lo":     hm["tail_mae_low"],
        "train_s":     round(elapsed, 1),
    })
    print(f"  {name}  val_mae={vm['mae']:.2f}  hold_mae={hm['mae']:.2f}  ({elapsed:.1f}s)")

# ── Print ranked table ────────────────────────────────────────────────────────

res = sorted(results, key=lambda r: r["hold_mae"])
best = res[0]["model"]

output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "model_comparison.csv"
json_path = output_dir / "model_comparison_summary.json"

pd.DataFrame(res).to_csv(csv_path, index=False)

summary = {
    "selected_model": best,
    "selection_criterion": "lowest holdout MAE",
    "n_candidates": len(res),
    "candidates": {r["model"]: r for r in res},
    "top_3_by_holdout_mae": res[:3],
}

json_path.write_text(json.dumps(summary, indent=2))

print("\n" + "=" * 85)
print(f"{'Model':<35} {'Val MAE':>8} {'Val RMSE':>9} {'Hold MAE':>9} {'Hold RMSE':>10} {'Tail Hi':>8} {'Train(s)':>9}")
print("-" * 85)
for r in res:
    marker = " ← BEST" if r["model"] == best else ""
    print(
        f"{r['model']:<35} {r['val_mae']:>8.2f} {r['val_rmse']:>9.2f} "
        f"{r['hold_mae']:>9.2f} {r['hold_rmse']:>10.2f} {r['tail_hi']:>8.2f} {r['train_s']:>9.1f}{marker}"
    )
print("=" * 85)
print(f"\nRecommended final model: {best}")
print(f"Saved CSV summary: {csv_path}")
print(f"Saved JSON summary: {json_path}")
print(
    "\nNext step: if a model beats LightGBM on holdout MAE by >1 EUR/MWh, "
    "consider updating src/models/extra_trees_model.py to use it. Otherwise keep LightGBM."
)
