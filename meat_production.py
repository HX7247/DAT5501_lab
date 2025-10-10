import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\meat-production-tonnes.csv"

def load_data(path=None):   
    path = path if path else file_path
    return pd.read_csv(path)

def plot_country_production_bar(df=None, top_n=None, show=True, save_path=None):
    df = df if df is not None else load_data()

    meat_cols = [c for c in df.columns if "Meat" in c]
    if not meat_cols:
        raise ValueError("No meat production column found in dataframe.")
    meat_col = meat_cols[0]

    df[meat_col] = pd.to_numeric(df[meat_col], errors="coerce")
    grouped = df.groupby("Entity", dropna=True)[meat_col].sum().reset_index()
    grouped = grouped.dropna(subset=["Entity"])
    grouped = grouped.sort_values(by=meat_col, ascending=False)

    if top_n:
        grouped = grouped.head(top_n)

    entities = grouped["Entity"].astype(str).values
    values = grouped[meat_col].values

    title_fs = 18
    label_fs = 14
    tick_fs = 12
    annot_fs = 11

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, max(len(entities), 1)))
    bars = plt.bar(entities, values, color=colors[:len(entities)], edgecolor="k")

    plt.title("Meat production by country", fontsize=title_fs)
    plt.ylabel("Production (tonnes)", fontsize=label_fs)
    plt.xticks(rotation=45, ha="right", fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)

    max_val = values.max() if len(values) else 0
    for bar, val in zip(bars, values):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + max_val * 0.01, f"{int(val):,}", 
                 ha="center", va="bottom", fontsize=annot_fs)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    df = load_data()
    plot_country_production_bar(df)