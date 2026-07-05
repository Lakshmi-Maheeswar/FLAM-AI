import time
import numpy as np
import pandas as pd

from solve import recover_parameters, full_residuals, Y0

DATA_PATH = "xy_data.csv"
N_RESTARTS = 10
NOISE_LEVELS = [0.0,0.001,0.01,0.05,0.1]  


def run_restarts(x, y, n_restarts=N_RESTARTS):
    thetas, Ms, Xs, runtimes, l1_means = [], [], [], [], []
    for seed in range(n_restarts):
        t0 = time.time()
        (theta, M, X), lm = recover_parameters(x, y, seed=seed)
        dt = time.time() - t0

        res = full_residuals((theta, M, X), x, y)
        thetas.append(np.rad2deg(theta))
        Ms.append(M)
        Xs.append(X)
        runtimes.append(dt)
        l1_means.append(np.mean(np.abs(res)))

    return map(np.array, (thetas, Ms, Xs, runtimes, l1_means))


def print_stats(label, thetas, Ms, Xs, runtimes, l1_means):
    print(f"\n=== {label}: {len(thetas)} Restarts ===")
    print(f"Mean theta       : {thetas.mean():.6f} deg")
    print(f"Std  theta       : {thetas.std():.3e} deg")
    print(f"Mean M           : {Ms.mean():.8f}")
    print(f"Std  M           : {Ms.std():.3e}")
    print(f"Mean X           : {Xs.mean():.6f}")
    print(f"Std  X           : {Xs.std():.3e}")
    print(f"Average Runtime  : {runtimes.mean():.3f} s")
    print(f"Average Mean L1  : {l1_means.mean():.6e}")


def main():
    df = pd.read_csv(DATA_PATH)
    x0, y0 = df["x"].values, df["y"].values


    thetas, Ms, Xs, runtimes, l1_means = run_restarts(x0, y0, N_RESTARTS)
    print_stats("Clean Data", thetas, Ms, Xs, runtimes, l1_means)


    if NOISE_LEVELS:
        print("\n=== Noise Robustness Study ===")
        rng = np.random.default_rng(0)
        for sigma in NOISE_LEVELS:
            th, Mm, Xx, rt, l1 = [], [], [], [], []
            for trial in range(5):
                xn = x0 + rng.normal(0, sigma, len(x0))
                yn = y0 + rng.normal(0, sigma, len(y0))
                t0 = time.time()
                (theta, M, X), lm = recover_parameters(xn, yn, seed=trial)
                dt = time.time() - t0
                res = full_residuals((theta, M, X), xn, yn)
                th.append(np.rad2deg(theta)); Mm.append(M); Xx.append(X)
                rt.append(dt); l1.append(np.mean(np.abs(res)))
            th, Mm, Xx = np.array(th), np.array(Mm), np.array(Xx)
            print(f"sigma={sigma:5.3f}  "
                  f"theta={th.mean():.5f}+/-{th.std():.1e} deg  "
                  f"M={Mm.mean():.6f}+/-{Mm.std():.1e}  "
                  f"X={Xx.mean():.5f}+/-{Xx.std():.1e}  "
                  f"runtime={np.mean(rt):.2f}s")


if __name__ == "__main__":
    main()