"""@author: Léa Chaccour & Ben Hasenson"""

# --- Force non-GUI backend before any Matplotlib import ---
import os

import matplotlib

matplotlib.use("Agg")
os.environ["MPLBACKEND"] = "Agg"

from feature_sets import FEATURE_SETS
from sklearn.model_selection import KFold, RandomizedSearchCV
from split_strategies import SplitMode, group_split, random_split
from utils.data_prep import ensure_output_dir, load_clean_data
from utils.metrics import evaluate_model, get_scorers
from utils.models import (
    TuningMode,
    apply_model_specific_scaling,
    get_model_dict,
    get_param_grid,
)
from utils.plotting import plot_relative_error_by_milestone
from utils.reporting import save_best_model, save_comparison_table

# ========== Configuration ==========
TUNING_MODE = TuningMode.HPC  # TuningMode.TEST or TuningMode.HPC
SPLIT_MODE = SplitMode.RANDOM # Choose split mode here: RANDOM or GROUP
K_FOLDS = 10  # Number of folds for cross-validation
N_ITER = (
    5 if TUNING_MODE == TuningMode.TEST else 50
)  # Number of random search iterations

# ========== Prepare data ==========
output_dir = ensure_output_dir("model_results")
clean_data = load_clean_data("data.csv")
target = "ITH"

# ========== Experiment Results Storage ==========
all_results = []
overall_best = {
    "rmse": float("inf"),
    "model_name": None,
    "feature_set_name": None,
    "fitted_model": None,
    "best_params": None,
}

# ========== Main Experiment Loop ==========
for fs_name, feature_list in FEATURE_SETS.items():
    print(f"\n\n============================")
    print(f"Running Feature Set: {fs_name.value}")
    print(f"Features: {feature_list}")
    print("============================")

    # --- Split Data ---
    if SPLIT_MODE == SplitMode.GROUP:
        X_train, X_test, y_train, y_test = random_split(
            clean_data, feature_list, target
        )
    else:
        X_train, X_test, y_train, y_test = group_split(clean_data, feature_list, target)

    # --- Loop over models ---
    for model_name, model in get_model_dict().items():
        print(f"\n--- Model: {model_name} | Feature Set: {fs_name.value} ---")

        param_grid = get_param_grid(model_name, TUNING_MODE)
        fit_X, test_X, fit_y, y_scaler = apply_model_specific_scaling(
            model_name, X_train, X_test, y_train
        )

        # --- Hyperparameter Search ---
        if param_grid:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=N_ITER,
                scoring=get_scorers(),
                refit="RMSE",
                cv=KFold(n_splits=K_FOLDS, shuffle=True, random_state=42),
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )
            search.fit(fit_X, fit_y)
            fitted_model = search.best_estimator_
            best_params = search.best_params_
            print("Best params:", best_params)
        else:
            model.fit(fit_X, fit_y)
            fitted_model = model
            best_params = model.get_params()

        # --- Predict & Evaluate ---
        y_pred_test = fitted_model.predict(test_X)
        y_pred_train = fitted_model.predict(fit_X)

        if y_scaler is not None:
            y_pred_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
            y_pred_train = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

        metrics = evaluate_model(y_test, y_pred_test)
        metrics.update(
            {
                "feature_set": fs_name.value,
                "model": model_name,
                "best_params": best_params,
            }
        )
        all_results.append(metrics)

        # --- Track Best Overall ---
        if metrics["RMSE"] < overall_best["rmse"]:
            overall_best.update(
                {
                    "rmse": metrics["RMSE"],
                    "model_name": model_name,
                    "feature_set_name": fs_name.value,
                    "fitted_model": fitted_model,
                    "best_params": best_params,
                }
            )

        # --- Plot Results ---
        milestone_train = clean_data.loc[X_train.index, "Milestone"]
        milestone_test = clean_data.loc[X_test.index, "Milestone"]

        plot_relative_error_by_milestone(
            y_train, y_pred_train, milestone_train,
            y_test, y_pred_test, milestone_test,
            model_name, output_dir
        )

# ========== Save and Summarize ==========
# Convert all_results into a comparison table (DataFrame)
save_comparison_table(
    output_dir,
    all_results,
    overall_best["model_name"],
    overall_best["fitted_model"],
    {r["model"]: r["best_params"] for r in all_results},
)

# Save best model
save_best_model(overall_best["fitted_model"], overall_best["model_name"], output_dir)

# Print summary
print("\n\n========== EXPERIMENT SUMMARY ==========")
print(f"Best Model:       {overall_best['model_name']}")
print(f"Best Feature Set: {overall_best['feature_set_name']}")
print(f"Lowest RMSE:      {overall_best['rmse']:.4f}")
print("Best Parameters:", overall_best["best_params"])
print("========================================")
