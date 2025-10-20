import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_CSV = Path(__file__).with_suffix("") .with_name(Path(__file__).stem).parent.joinpath("quadratic_data.csv")
DEFAULT_PLOT = Path(__file__).with_suffix("") .with_name(Path(__file__).stem).parent.joinpath("quadratic_fit.png")

def generate_quadratic_data(a=1.0, b=0.0, c=0.0, x_min=-10, x_max=10, n=100, noise_std=10.0, seed=0, out_csv: Path | str = None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, n)
    x.sort()
    y_true = a * x**2 + b * x + c
    noise = rng.normal(0, noise_std, size=n)
    y = y_true + noise
    df = pd.DataFrame({"x": x, "y": y, "y_true": y_true})
    out = Path(out_csv) if out_csv else DEFAULT_CSV
    df.to_csv(out, index=False)
    return df, out

def fit_quadratic_and_stats(df: pd.DataFrame):
    x = df["x"].values
    y = df["y"].values

    if x.size < 2:
        raise ValueError("Need at least 2 points to fit.")
    coeffs = np.polyfit(x, y, 2)  
    y_pred = np.polyval(coeffs, x)

    if np.allclose(y_pred, y_pred[0]):
        r = np.nan
    else:
        r = np.corrcoef(y, y_pred)[0, 1]
    r2 = None if np.isnan(r) else r**2
    return coeffs, y_pred, r, r2

def plot_data_and_fit(df: pd.DataFrame, y_pred=None, out_png: Path | str = None, show=True):
    x = df["x"].values
    y = df["y"].values
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, s=30, alpha=0.7, label="data")
    if y_pred is not None:
        order = np.argsort(x)
        plt.plot(x[order], y_pred[order], color="red", linestyle="--", linewidth=2, label="quadratic fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Quadratic fit to generated data")
    plt.legend()
    plt.grid(True)
    out = Path(out_png) if out_png else DEFAULT_PLOT
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return out

if __name__ == "__main__":
    df, csv_path = generate_quadratic_data(a=2.0, b=-1.0, c=5.0, x_min=-5, x_max=5, n=80, noise_std=8.0, seed=42)
    coeffs, y_pred, r, r2 = fit_quadratic_and_stats(df)
    print(f"Data saved to: {csv_path}")
    print(f"Fitted coefficients (a, b, c): {coeffs[0]:.6f}, {coeffs[1]:.6f}, {coeffs[2]:.6f}")
    if np.isnan(r):
        print("Correlation R: NaN (constant prediction)")
    else:
        print(f"Correlation R: {r:.4f}")
        print(f"R-squared: {r2:.4f}")
    png_path = plot_data_and_fit(df, y_pred=y_pred)
    print(f"Plot saved to: {png_path}")