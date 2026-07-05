

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

DATA_PATH = "xy_data.csv"
Y0 = 42.0                      # known additive offset in y(t)
BOUNDS_THETA = (np.deg2rad(0.01), np.deg2rad(49.99))
BOUNDS_X = (0.01, 99.99)


# ---------------------------------------------------------------------------
# Rotation reformulation
# ---------------------------------------------------------------------------
# (x - X, y - 42) is a rigid rotation R(theta) of the canonical point
# (t, v(t)), v(t) = exp(M|t|) sin(0.3t). Since R is orthogonal, the inverse
# rotation R(theta)^T recovers (t, v) EXACTLY for any candidate (theta, X) --
# this is linear algebra, not optimization, and perfectly conditioned.
def recover_tv(x, y, theta, X):
    c, s = np.cos(theta), np.sin(theta)
    xp, yp = x - X, y - Y0
    t = xp * c + yp * s
    v = -xp * s + yp * c
    return t, v


def fit_M_closed_form(t, v, eps=1e-9):
    s03 = np.sin(0.3 * t)
    pos = v * s03 > 0               
    if pos.sum() < 20:
        return 0.0
    w = s03[pos] ** 2             
    ratio = v[pos] / s03[pos]
    z = np.log(np.abs(ratio) + eps)
    tt = t[pos]
    M = np.sum(w * tt * z) / np.sum(w * tt * tt)
    return M


def concentrated_sse(params, x, y):
    theta, X = params
    t, v = recover_tv(x, y, theta, X)
    M = fit_M_closed_form(t, v)
    pred = np.exp(np.clip(M * t, -50, 50)) * np.sin(0.3 * t)
    return np.sum((v - pred) ** 2)


def full_residuals(params, x, y):
    theta, M, X = params
    t, v = recover_tv(x, y, theta, X)
    pv = np.exp(np.clip(M * np.abs(t), -50, 50)) * np.sin(0.3 * t)
    x_pred = t * np.cos(theta) - pv * np.sin(theta) + X
    y_pred = Y0 + t * np.sin(theta) + pv * np.cos(theta)
    return np.concatenate([x_pred - x, y_pred - y])


def recover_parameters(x, y, seed=42):
    bounds = [BOUNDS_THETA, BOUNDS_X]
    de = differential_evolution(
        concentrated_sse, bounds, args=(x, y),
        seed=seed, tol=1e-14, maxiter=300, popsize=25, polish=True,
    )
    theta0, X0 = de.x
    t0, v0 = recover_tv(x, y, theta0, X0)
    M0 = fit_M_closed_form(t0, v0)

    lm = least_squares(
        full_residuals, [theta0, M0, X0], args=(x, y),
        method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15,
    )
    return lm.x, lm


def main():
    df = pd.read_csv(DATA_PATH)
    x, y = df["x"].values, df["y"].values

    (theta, M, X), lm = recover_parameters(x, y)
    residuals = full_residuals((theta, M, X), x, y)
    l1_total = np.sum(np.abs(residuals))
    l1_mean = np.mean(np.abs(residuals))
    l1_max = np.max(np.abs(residuals))

    print("=== Recovered Parameters ===")
    print(f"theta = {np.rad2deg(theta):.6f} deg")
    print(f"M     = {M:.8f}")
    print(f"X     = {X:.6f}")
    print()
    print("=== Reconstruction Error ===")
    print(f"L1 total  = {l1_total:.6e}")
    print(f"L1 mean   = {l1_mean:.6e}")
    print(f"L1 max    = {l1_max:.6e}")

    # Save recovered parameters
    np.save("best_params.npy", np.array([theta, M, X]))
    with open("best_params.txt", "w") as f:
        f.write(f"theta_rad={theta!r}\n")
        f.write(f"theta_deg={np.rad2deg(theta)!r}\n")
        f.write(f"M={M!r}\n")
        f.write(f"X={X!r}\n")
        f.write(f"L1_total={l1_total!r}\n")
        f.write(f"L1_mean={l1_mean!r}\n")
        f.write(f"L1_max={l1_max!r}\n")

    print("\nSaved: best_params.npy, best_params.txt")


if __name__ == "__main__":
    main()