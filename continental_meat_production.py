import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Define file path to meat production data
file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\global-meat-production.csv"

def load_data(path=None):   
    # Use provided path or default file path
    path = path if path else file_path
    # Load CSV file into DataFrame
    return pd.read_csv(path)

def plot_continent_production_bar(df, show=True, save_path=None):
    # Find columns containing "Meat" in name
    meat_cols = [c for c in df.columns if "Meat" in c]
    # Raise error if no meat column found
    if not meat_cols:
        raise ValueError("No meat production column found in dataframe.")
    # Use first meat column found
    meat_col = meat_cols[0]

    # Convert meat production values to numeric type
    df[meat_col] = pd.to_numeric(df[meat_col], errors="coerce")
    # Group by entity and sum total production
    grouped = df.groupby("Entity", dropna=True)[meat_col].sum().reset_index()

    # Remove rows with missing entity names
    grouped = grouped.dropna(subset=["Entity"])
    # Sort by production value in descending order
    grouped = grouped.sort_values(by=meat_col, ascending=False)

    # Extract entity names and production values
    entities = grouped["Entity"].astype(str).values
    values = grouped[meat_col].values

    # Define font sizes for different elements
    title_fs = 18
    label_fs = 16
    tick_fs = 14
    annot_fs = 12

    # Generate color palette for bars
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, max(len(entities), 1)))

    # Create figure and bar chart
    plt.figure(figsize=(12, 7))
    bars = plt.bar(entities, values, color=colors[:len(entities)], edgecolor="k")
    # Add axis labels and title
    plt.ylabel("Meat production (hundred million tonnes)", fontsize=label_fs)
    plt.title("Total meat production by continent", fontsize=title_fs)
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=tick_fs)
    plt.tick_params(axis='y', labelsize=tick_fs)

    # Add value labels on top of each bar
    max_val = values.max() if len(values) else 0
    for bar, val in zip(bars, values):
        h = bar.get_height()
        # Place text above bar with formatted value
        plt.text(bar.get_x() + bar.get_width()/2, h + max_val*0.01, f"{val:,.0f}", 
                 ha="center", va="bottom", fontsize=annot_fs)

    # Adjust layout and save if path provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # Display or close plot
    if show:
        plt.show()
    else:
        plt.close()

# Main execution block
if __name__ == "__main__":
    # Load data and create visualization
    df = load_data()
    plot_continent_production_bar(df)