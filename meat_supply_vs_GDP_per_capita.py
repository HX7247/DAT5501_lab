import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Default dataset path
file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\Meat Supply Vs GDP Per capita.csv"

def load_data(path=None):   
    # Load CSV file, using default path if none provided
    path = path if path else file_path
    return pd.read_csv(path)

def plot_meat_vs_gdp(df, show=True, save_path=None):
    # Auto-detect relevant column names
    meat_col = next((c for c in df.columns if "Meat" in c), None)
    gdp_col = next((c for c in df.columns if "GDP per capita" in c), None)
    region_col = next((c for c in df.columns if "world regions" in c.lower() 
                       or "region" in c.lower() or "continent" in c.lower()), None)

    # Ensure essential columns exist
    if meat_col is None or gdp_col is None:
        raise ValueError("Required columns not found. Available columns: " + ", ".join(df.columns))

    # Extract needed columns into new DataFrame
    df_plot = df[[meat_col, gdp_col] + ([region_col] if region_col else [])].copy()

    # Convert numeric columns
    df_plot[meat_col] = pd.to_numeric(df_plot[meat_col], errors="coerce")
    df_plot[gdp_col] = pd.to_numeric(df_plot[gdp_col], errors="coerce")

    # Replace missing region labels
    if region_col:
        df_plot[region_col] = df_plot[region_col].fillna("Unknown")

    # Drop rows missing critical values
    df_plot = df_plot.dropna(subset=[meat_col, gdp_col])

    # Extract arrays for plotting
    x = df_plot[gdp_col].values
    y = df_plot[meat_col].values

    plt.figure(figsize=(10, 7))

    region_patches = []   # For legend entries
    line_handle = None    # For regression line legend
    fit_exists = False    # Track whether regression was plotted

    # If region info exists, color points by region
    if region_col:
        regions = list(df_plot[region_col].unique())
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, len(regions)))
        color_map = dict(zip(regions, colors))
        point_colors = df_plot[region_col].map(color_map)

        plt.scatter(x, y, c=list(point_colors), alpha=0.8, edgecolors='k', linewidths=0.3)

        # Create colored legend patches
        region_patches = [mpatches.Patch(color=color_map[r], label=r) for r in regions]
    else:
        plt.scatter(x, y, alpha=0.8, edgecolors='k', linewidths=0.3)

    # Axis labels and title
    plt.xlabel("GDP per capita")
    plt.ylabel("Meat Supply (kg per capita per year)")
    plt.title("Meat per capita vs GDP per capita")
    plt.grid(True)

    # Add linear regression line if enough data
    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)                     # slope & intercept
        x_sorted = np.linspace(x.min(), x.max(), 100)    # smooth line
        plt.plot(x_sorted, np.polyval(coeffs, x_sorted),
                 color='red', linestyle='--', label='linear fit')

        fit_exists = True
        line_handle = Line2D([0], [0], color='red', linestyle='--', label='linear fit')

    # Build legend depending on whether regions exist
    if region_col:
        handles = region_patches.copy()
        if fit_exists:
            handles.append(line_handle)
        plt.legend(handles=handles, title="Region/ Continent",
                   bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        if fit_exists:
            plt.legend(loc="lower right")

    plt.tight_layout()

    # Optionally save the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show or close plot
