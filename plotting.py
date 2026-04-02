"""@author: Ben Hasenson & Léa Chaccours"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import os
def plot_regression_results(
    y_true: pd.Series, y_pred: pd.Series, model_name: str, output_dir: str
):
    """
    Creates and saves a scatter plot of predicted vs actual values,
    with color-coded error and annotated metrics.
    """

    # Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    abs_error = np.abs(y_true - y_pred)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        y_true, y_pred, c=abs_error, cmap="Blues", edgecolor="k", s=140
    )
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "orange", lw=2)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.title(f"Regression Results: {model_name}")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Deviation")

    # Annotated box with metrics
    textstr = f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}"
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=20,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"regression_plot_{model_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved regression plot: {plot_path}")



# ========== Relative error plot function ==========
def plot_relative_error(y_true, y_pred, model_name, output_dir):
    """Plots the relative error between true and predicted values."""

    rel_error = (y_true - y_pred) / y_true

    plt.figure(figsize=(8, 5))
    plt.scatter(
        y_true,
        rel_error,
        color="blue",
        alpha=0.6,
        edgecolor="k",
        s=140,
        label="Data points",
    )
    plt.axhline(0, color="red", linestyle="--", lw=2, label="Zero error")

    # Labels and title with I_th in subscript
    plt.xlabel(
        r"$I_{th}$ (Real)", fontsize=20, fontweight="bold", fontname="Times New Roman"
    )
    plt.ylabel(
        r"Relative Error $(I_{th}^{real} - I_{th}^{pred}) / I_{th}^{real}$",
        fontsize=20,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.title(
        r"Relative Error vs Real $I_{th}$: " + model_name,
        fontsize=22,
        fontweight="bold",
        fontname="Times New Roman",
    )

    # Axis tick labels
    plt.xticks(fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.yticks(fontsize=20, fontweight="bold", fontname="Times New Roman")

    # Legend
    plt.legend(prop={"size": 20, "weight": "bold", "family": "Times New Roman"})

    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"relative_error_{model_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved relative error plot: {plot_path}")

def plot_relative_error_by_milestone(
    y_train, y_pred_train, milestone_train,
    y_test, y_pred_test, milestone_test,
    model_name, output_dir
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # =========================
    # Compute relative errors
    # =========================
    rel_error_train = (y_pred_train - y_train) / y_train
    rel_error_test = (y_pred_test - y_test) / y_test

    # =========================
    # Compute global axis limits
    # =========================
    # Y-axis: relative errors
    y_min_raw = min(rel_error_train.min(), rel_error_test.min())
    y_max_raw = max(rel_error_train.max(), rel_error_test.max())
    y_range = y_max_raw - y_min_raw
    y_min = y_min_raw - 0.05 * y_range  # 5% buffer
    y_max = y_max_raw + 0.05 * y_range

    # X-axis: real values
    x_min_raw = min(y_train.min(), y_test.min())
    x_max_raw = max(y_train.max(), y_test.max())
    x_range = x_max_raw - x_min_raw
    x_min = x_min_raw - 0.05 * x_range  # 5% buffer
    x_max = x_max_raw + 0.05 * x_range

    # =========================
    # Colors for milestones
    # =========================
    all_milestones = np.unique(np.concatenate((milestone_train, milestone_test)))
    cmap = plt.get_cmap("tab10")
    milestone_color_map = {m: cmap(i % 10) for i, m in enumerate(all_milestones)}

    # =========================
    # Plot: Train
    # =========================
    plt.figure(figsize=(10, 6))
    for milestone in all_milestones:
        mask = milestone_train == milestone
        plt.scatter(
            y_train[mask],
            rel_error_train[mask],
            color=milestone_color_map[milestone],
            edgecolor='k',
            s=140,
            alpha=0.8,
            label=fr"$t_{{stress}}$ {milestone}",
            clip_on=False  # ensures big markers are not clipped
        )
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel(r"$I_{th}$ (Real)", fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.ylabel(r"Relative Error $(I_{th}^{pred} - I_{th}^{real}) / I_{th}^{real}$",
               fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.title(rf"Train Set: Relative Error vs Real $I_{{th}}$ by Milestone – {model_name}",
              fontsize=22, fontweight="bold", fontname="Times New Roman")
    plt.xticks(fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.yticks(fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(prop={'size': 20, 'weight': 'bold', 'family': 'Times New Roman'})
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"relative_error_by_milestone_{model_name}.png")
    plt.tight_layout()
    plt.savefig(train_path, dpi=300)
    plt.close()
    print(f"Saved Train plot: {train_path}")

    # =========================
    # Plot: Test
    # =========================
    plt.figure(figsize=(10, 6))
    for milestone in np.unique(milestone_test):
        mask = milestone_test == milestone
        plt.scatter(
            y_test[mask],
            rel_error_test[mask],
            color=milestone_color_map[milestone],
            edgecolor='k',
            s=140,
            alpha=0.8,
            label=fr"$t_{{stress}}$ {milestone}",
            clip_on=False
        )
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel(r"$I_{th}$ (Real)", fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.ylabel(r"Relative Error $(I_{th}^{pred} - I_{th}^{real}) / I_{th}^{real}$",
               fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.title(f"Test Set: Relative Error vs Real $I_{{th}}$ by Milestone – {model_name}",
              fontsize=22, fontweight="bold", fontname="Times New Roman")
    plt.xticks(fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.yticks(fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(prop={'size': 20, 'weight': 'bold', 'family': 'Times New Roman'})
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    test_path = os.path.join(output_dir, f"figure2_test_relative_error_{model_name}.png")
    plt.tight_layout()
    plt.savefig(test_path, dpi=300)
    plt.close()
    print(f"Saved Test plot: {test_path}")
