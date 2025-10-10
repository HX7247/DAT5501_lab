import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\Meat Supply Vs GDP Per capita.csv"

def load_data(path=None):   
    path = path if path else file_path
    return pd.read_csv(path)

def _regression_stats(x, y):
    if len(x) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    r2 = r ** 2
    return slope, intercept, r, r2

def plot_meat_vs_gdp(df, show=True, save_path=None):
    meat_col = next((c for c in df.columns if "Meat" in c), None)
    gdp_col = next((c for c in df.columns if "GDP per capita" in c), None)
    region_col = next((c for c in df.columns if "world regions" in c.lower() or "region" in c.lower() or "continent" in c.lower()), None)

    if meat_col is None or gdp_col is None:
        raise ValueError("Required columns not found. Available columns: " + ", ".join(df.columns))

    df_plot = df[[meat_col, gdp_col] + ([region_col] if region_col else [])].copy()
    df_plot[meat_col] = pd.to_numeric(df_plot[meat_col], errors="coerce")
    df_plot[gdp_col] = pd.to_numeric(df_plot[gdp_col], errors="coerce")
    if region_col:
        df_plot[region_col] = df_plot[region_col].fillna("Unknown")
    df_plot = df_plot.dropna(subset=[meat_col, gdp_col])

    x = df_plot[gdp_col].values
    y = df_plot[meat_col].values

    plt.figure(figsize=(10, 7))

    region_patches = []
    line_handle = None
    fit_exists = False

    if region_col:
        regions = list(df_plot[region_col].unique())
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, len(regions)))
        color_map = dict(zip(regions, colors))
        point_colors = df_plot[region_col].map(color_map)

        plt.scatter(x, y, c=list(point_colors), alpha=0.8, edgecolors='k', linewidths=0.3)
        region_patches = [mpatches.Patch(color=color_map[r], label=r) for r in regions]
    else:
        plt.scatter(x, y, alpha=0.8, edgecolors='k', linewidths=0.3)

    plt.xlabel("GDP per capita")
    plt.ylabel("Meat Supply (kg per capita per year)")
    plt.title("Meat per capita vs GDP per capita")
    plt.grid(True)

    slope, intercept, r, r2 = _regression_stats(x, y)
    if not np.isnan(slope):
        x_sorted = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_sorted, np.polyval((slope, intercept), x_sorted), color='red', linestyle='--', label='linear fit')
        fit_exists = True
        line_handle = Line2D([0], [0], color='red', linestyle='--', label='linear fit')

        stats_text = f"Overall\nR = {r:.3f}\nR² = {r2:.3f}\nSlope = {slope:.4f}"
        plt.gca().text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9)

        print("Overall regression:")
        print(f"  slope = {slope:.6f}, intercept = {intercept:.6f}, R = {r:.6f}, R^2 = {r2:.6f}")

    if region_col:
        per_region_texts = []
        for reg in regions:
            sel = df_plot[df_plot[region_col] == reg]
            xr = sel[gdp_col].values
            yr = sel[meat_col].values
            if len(xr) >= 2:
                s, itc, rr, rr2 = _regression_stats(xr, yr)
                per_region_texts.append(f"{reg}: R={rr:.3f}, R²={rr2:.3f}")
            else:
                per_region_texts.append(f"{reg}: n={len(xr)} (insufficient for R)")

        print("Per-region regression stats:")
        for t in per_region_texts:
            print("  " + t)

        stats_block = "\n".join(per_region_texts)
        plt.gca().text(0.98, 0.22, stats_block, transform=plt.gca().transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)

    if region_col:
        handles = region_patches.copy()
        if fit_exists:
            handles.append(line_handle)
        plt.legend(handles=handles, title="Region/ Continent", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        if fit_exists:
            plt.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_region_averages(df, show=True, save_path=None):
    meat_col = next((c for c in df.columns if "Meat" in c), None)
    gdp_col = next((c for c in df.columns if "GDP per capita" in c), None)
    region_col = next((c for c in df.columns if "world regions" in c.lower() or "region" in c.lower() or "continent" in c.lower()), None)

    if meat_col is None or gdp_col is None or region_col is None:
        raise ValueError("Required columns (meat, gdp, region) not found. Available: " + ", ".join(df.columns))

    df_plot = df[[meat_col, gdp_col, region_col]].copy()
    df_plot[meat_col] = pd.to_numeric(df_plot[meat_col], errors="coerce")
    df_plot[gdp_col] = pd.to_numeric(df_plot[gdp_col], errors="coerce")
    df_plot[region_col] = df_plot[region_col].fillna("Unknown")
    df_plot = df_plot.dropna(subset=[meat_col, gdp_col])

    region_means = df_plot.groupby(region_col)[[gdp_col, meat_col]].mean().reset_index()

    regions = list(region_means[region_col])
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(regions)))
    color_map = dict(zip(regions, colors))

    x_all = df_plot[gdp_col].values
    y_all = df_plot[meat_col].values

    plt.figure(figsize=(10, 7))
    plt.scatter(x_all, y_all, color="#cccccc", alpha=0.5, s=30, edgecolors='none', label="Countries")

    x_mean = region_means[gdp_col].values
    y_mean = region_means[meat_col].values
    mean_colors = [color_map[r] for r in region_means[region_col]]

    plt.scatter(x_mean, y_mean, c=mean_colors, s=200, edgecolors='k', linewidths=0.8)
    for xi, yi, r in zip(x_mean, y_mean, region_means[region_col]):
        plt.text(xi, yi, "  " + str(r), verticalalignment='center', fontsize=9)

    s, itc, rr, rr2 = _regression_stats(x_mean, y_mean)
    if not np.isnan(s):
        x_sorted = np.linspace(x_mean.min(), x_mean.max(), 100)
        plt.plot(x_sorted, np.polyval((s, itc), x_sorted), color='red', linestyle='--', label='mean linear fit')
        stats_text = f"Region means\nR = {rr:.3f}\nR² = {rr2:.3f}\nSlope = {s:.4f}"
        plt.gca().text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9)
        print("Region-means regression:")
        print(f"  slope = {s:.6f}, intercept = {itc:.6f}, R = {rr:.6f}, R^2 = {rr2:.6f}")

    plt.xlabel("GDP per capita")
    plt.ylabel("Meat Supply (kg per capita per year)")
    plt.title("Region mean: Meat supply vs GDP per capita")
    plt.grid(True)

    region_patches = [mpatches.Patch(color=color_map[r], label=r) for r in regions]
    country_handle = Line2D([0], [0], marker='o', color='w', label='Countries', markerfacecolor='#cccccc', markersize=6)
    handles = [country_handle] + region_patches
    plt.legend(handles=handles, title="Region/ Continent", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    df = load_data()
    plot_meat_vs_gdp(df)
    plot_region_averages(df)