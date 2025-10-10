import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\global-meat-production.csv"

def load_data(path=None):   
    path = path if path else file_path
    return pd.read_csv(path)
