import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define default output paths for CSV and PNG files
DEFAULT_CSV = Path(__file__).with_suffix("").with_name(Path(__file__).stem).parent.joinpath("quadratic_data.csv")
DEFAULT_PLOT = Path(__file__).with_suffix("").with_name(Path(__file__).stem).parent.joinpath("quadratic_fit.png")

def generate_quadratic_data(a=1.0, b=0.0, c=0.0, x_min=-10, x_max=10, n=100, noise_std=10.0, seed=0, out_csv: Path | str = None):
    """Generate synthetic quadratic data with noise and save to CSV"""
    # Initialize random number generator with seed for reproducibility
    rng = np.random.default_rng(seed)
    # Generate uniformly distributed x values and sort them
    x = rng.uniform(x_min, x_max, n)
    x.sort()
    # Calculate true y values from quadratic equation y = ax^2 + bx + c
    y_true = a * x**2 + b * x + c
    # Add Gaussian noise to y values
    noise = rng.normal(0, noise_std, size=n)
    y = y_true + noise
    # Create DataFrame with x, y, and true y values
    df = pd.DataFrame({"x": x, "y": y, "y_true": y_true})
    # Save to CSV file
    out = Path(out_csv) if out_csv else DEFAULT_CSV
    df.to_csv(out, index=False)
    return df, out

def fit_quadratic_and_stats(df: pd.DataFrame):
    """Fit a quadratic polynomial to data and calculate correlation statistics"""
    # Extract x and y columns from DataFrame
    x = df["x"].values
    y = df["y"].values

    # Validate minimum data points required
    if x.size < 2:
        raise ValueError("Need at least 2 points to fit.")
    # Fit 2nd degree polynomial (quadratic)
    coeffs = np.polyfit(x, y, 2)  
    # Generate predicted y values using fitted coefficients
    y_pred = np.polyval(coeffs, x)

    # Calculate correlation coefficient (R), handling edge case of constant predictions
    if np.allclose(y_pred, y_pred[0]):
        r = np.nan
    else:
        r = np.corrcoef(y, y_pred)[0, 1]
    # Calculate R-squared (coefficient of determination)
    r2 = None if np.isnan(r) else r**2
    return coeffs, y_pred, r, r2

def plot_data_and_fit(df: pd.DataFrame, y_pred=None, out_png: Path | str = None, show=True):
    """Create scatter plot of data with fitted quadratic curve"""
    # Extract x and y data
    x = df["x"].values
    y = df["y"].values
    # Create figure with specified size
    plt.figure(figsize=(8,6))
    # Plot data points as scatter
    plt.scatter(x, y, s=30, alpha=0.7, label="data")
    # Plot fitted curve if predictions provided
    if y_pred is not None:
        order = np.argsort(x)
        plt.plot(x[order], y_pred[order], color="red", linestyle="--", linewidth=2, label="quadratic fit")
    # Add labels, title, and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Quadratic fit to generated data")
    plt.legend()
    plt.grid(True)
    # Save plot to PNG file
    out = Path(out_png) if out_png else DEFAULT_PLOT
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    # Display or close plot based on show parameter
    if show:
        plt.show()
    else:
        plt.close()
    return out

if __name__ == "__main__":
    # Generate synthetic quadratic data with specific parameters
    df, csv_path = generate_quadratic_data(a=2.0, b=-1.0, c=5.0, x_min=-5, x_max=5, n=80, noise_std=8.0, seed=42)
    # Fit quadratic model and calculate statistics
    coeffs, y_pred, r, r2 = fit_quadratic_and_stats(df)
    # Print results
    print(f"Data saved to: {csv_path}")
    print(f"Fitted coefficients (a, b, c): {coeffs[0]:.6f}, {coeffs[1]:.6f}, {coeffs[2]:.6f}")
    if np.isnan(r):
        print("Correlation R: NaN (constant prediction)")
    else:
        print(f"Correlation R: {r:.4f}")
        print(f"R-squared: {r2:.4f}")
    # Generate and save plot
    png_path = plot_data_and_fit(df, y_pred=y_pred)
    print(f"Plot saved to: {png_path}")