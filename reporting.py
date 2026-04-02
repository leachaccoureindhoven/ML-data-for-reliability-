import os
import pandas as pd
import joblib

def save_best_model(best_model, best_model_name: str, output_dir: str):
    """Saves the best model object to disk."""
    best_model_file = os.path.join(output_dir, f"best_model_{best_model_name}.joblib")
    joblib.dump(best_model, best_model_file)
    print(f"✅ Saved best model object to: {best_model_file}")


def save_comparison_table(output_dir, results_list, best_model_name, best_model, best_params_dict):
    """
    Saves a detailed comparison of all model-feature-set combinations, 
    their metrics, and best hyperparameters.
    """
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(results_list)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ---- Sort by RMSE (ascending = better) ----
    df_sorted = df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    df_sorted["Rank"] = df_sorted.index + 1

    # ---- Save as CSV for quantitative analysis ----
    csv_path = os.path.join(output_dir, "model_feature_comparison.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"Saved full model-feature comparison CSV to: {csv_path}")

    # ---- Also save a readable text summary ----
    txt_path = os.path.join(output_dir, "model_scores_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=== Model & Feature Set Comparison Summary ===\n\n")

        # Write overall best
        best_row = df_sorted.iloc[0]
        f.write(f"Best Overall Combination:\n")
        f.write(f"  Model:        {best_row['model']}\n")
        f.write(f"  Feature Set:  {best_row['feature_set']}\n")
        f.write(f"  RMSE:         {best_row['RMSE']:.4f}\n")
        f.write(f"  R2:           {best_row['R2']:.4f}\n")
        f.write(f"  MAE:          {best_row['MAE']:.4f}\n")
        f.write(f"  MAPE:         {best_row['MAPE']:.4f}\n")
        f.write("\n" + "=" * 60 + "\n\n")

        # ---- Best Model per Feature Set ----
        f.write("Best Model per Feature Set:\n")
        grouped = df_sorted.groupby("feature_set").first().reset_index()
        for _, row in grouped.iterrows():
            f.write(
                f"  {row['feature_set']:<25} -> {row['model']:<15} | "
                f"RMSE: {row['RMSE']:.4f} | R2: {row['R2']:.4f}\n"
            )
        f.write("\n" + "=" * 60 + "\n\n")

        # ---- Top 5 Overall Leaderboard ----
        f.write("Top 5 Overall Combinations:\n")
        top5 = df_sorted.head(5)
        for _, row in top5.iterrows():
            f.write(
                f"  Rank {row['Rank']:>2}: {row['model']:<15} "
                f"+ {row['feature_set']:<25} | RMSE: {row['RMSE']:.4f}\n"
            )

        f.write("\n" + "=" * 60 + "\n\n")

        # ---- Detailed Metrics Table ----
        f.write(f"{'Rank':<5} {'Model':<15} {'FeatureSet':<25} {'R2':>8} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}\n")
        f.write("-" * 80 + "\n")
        for _, row in df_sorted.iterrows():
            f.write(
                f"{row['Rank']:<5} {row['model']:<15} {row['feature_set']:<25} "
                f"{row['R2']:>8.4f} {row['RMSE']:>10.4f} {row['MAE']:>10.4f} {row['MAPE']:>10.4f}\n"
            )

        # ---- Hyperparameter Summary ----
        f.write("\n\nBest Hyperparameters per Model:\n")
        for name, params in best_params_dict.items():
            f.write(f"\n{name}:\n")
            for key, val in params.items():
                f.write(f"  {key}: {val}\n")

    print(f"Saved formatted comparison summary to: {txt_path}")
