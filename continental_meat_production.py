import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\global-meat-production.csv"

def load_data(path=None):   
    path = path if path else file_path
    return pd.read_csv(path)

def plot_continent_production_bar(df, show=True, save_path=None):
    """
    Aggregate meat production by continent (Entity) and plot a bar chart.
    """
    # find a meat production column by substring (handles duplicate column names)
    meat_cols = [c for c in df.columns if "Meat" in c]
    if not meat_cols:
        raise ValueError("No meat production column found in dataframe.")
    meat_col = meat_cols[0]

    # ensure numeric and group by Entity
    df[meat_col] = pd.to_numeric(df[meat_col], errors="coerce")
    grouped = df.groupby("Entity", dropna=True)[meat_col].sum().reset_index()

    # drop NaN/empty Entity rows
    grouped = grouped.dropna(subset=["Entity"])
    grouped = grouped.sort_values(by=meat_col, ascending=False)

    entities = grouped["Entity"].astype(str).values
    values = grouped[meat_col].values

    # color map
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, max(len(entities), 1)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(entities, values, color=colors[:len(entities)], edgecolor="k")
    plt.ylabel("Meat production (hundred million tonnes)")
    plt.title("Total meat production by continent")
    plt.xticks(rotation=45, ha="right")

    # annotate bars
    max_val = values.max() if len(values) else 0
    for bar, val in zip(bars, values):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + max_val*0.01, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    df = load_data()
    plot_continent_production_bar(df)