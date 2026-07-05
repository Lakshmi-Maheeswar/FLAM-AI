

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "xy_data.csv"
PARAMS_PATH = "best_params.npy"
OUT_DIR = "results"
OUT_PATH = os.path.join(OUT_DIR, "fit_validation.png")
Y0 = 42.0


def curve_xy(t, theta, M, X):
    v = np.exp(np.clip(M * np.abs(t), -50, 50)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - v * np.sin(theta) + X
    y = Y0 + t * np.sin(theta) + v * np.cos(theta)
    return x, y


def main():
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(
            f"{PARAMS_PATH} not found -- run solve.py first to recover parameters."
        )

    theta, M, X = np.load(PARAMS_PATH)

    df = pd.read_csv(DATA_PATH)
    x, y = df["x"].values, df["y"].values

    t_dense = np.linspace(6, 60, 2000)
    x_fit, y_fit = curve_xy(t_dense, theta, M, X)

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=6, c="#4C72B0", alpha=0.5, label="Observed data (xy_data.csv)")
    ax.plot(x_fit, y_fit, c="#DD8452", lw=1.5,
            label=f"Recovered curve (θ={np.rad2deg(theta):.3f}°, "
                  f"M={M:.5f}, X={X:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fit Validation: Recovered Curve vs Observed Data")
    ax.legend()
    ax.set_aspect("equal", "box")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()