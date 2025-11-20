# DAT5501_lab

## Project Overview
This repository contains various Python scripts for data analysis, sorting complexity studies, and utility programs.

---

## Code Documentation

### 1. **daily_price_change_sorting_activity.py**
**Purpose:** Analyzes sorting time complexity by measuring how long it takes to sort daily stock price changes for different array sizes.

**What it does:**
- Loads NVIDIA historical stock data
- Calculates daily price changes (ΔP = P_{n+1} - P_n)
- Measures sorting time for array sizes from 10 to 100,000
- Plots measured sorting times against theoretical O(n log n) complexity
- Calculates correlation coefficient between measured and theoretical times

**Dependencies:**
- `numpy`
- `pandas`
- `matplotlib`

**Input Data:** `data/HistoricalData_NVDA.csv`

**Output:** Plot showing sorting time complexity analysis

---

### 2. **Duration_calculator.py**
**Purpose:** Calculates the number of days until/since a target date in ISO format (YYYY-MM-DD).

**What it does:**
- Takes user input for a target date
- Calculates days remaining until that date
- Handles edge cases (today, past dates)
- Validates input format

**Dependencies:**
- `datetime` (built-in)
- `unittest` (built-in)

**Usage:** Run the script and enter a date in YYYY-MM-DD format

---

### 3. **calendar_printer.py**
**Purpose:** Generates an ASCII calendar for a given month and starting day.

**What it does:**
- Prompts user for number of days in month (28-31)
- Prompts user for starting day of the week
- Displays formatted calendar grid

**Dependencies:**
- None (uses built-in functions)

**Usage:** Run the script and provide month length and starting day

---

### 4. **advanced_data_fitting.py**
**Purpose:** Generates quadratic data with noise and performs polynomial fitting with statistical analysis.

**What it does:**
- Generates synthetic quadratic data (y = ax² + bx + c) with configurable noise
- Saves generated data to CSV
- Fits a quadratic polynomial to the data
- Calculates correlation coefficient and R² value
- Creates visualization with fitted curve

**Dependencies:**
- `numpy`
- `pandas`
- `matplotlib`
- `pathlib` (built-in)

**Output Files:**
- `quadratic_data.csv` (generated data)
- `quadratic_fit.png` (plot)

---

### 5. **continental_meat_production.py**
**Purpose:** Analyzes and visualizes total meat production by continent.

**What it does:**
- Loads global meat production data
- Aggregates production by continent
- Sorts by total production
- Creates bar chart with annotations

**Dependencies:**
- `numpy`
- `pandas`
- `matplotlib`
- `matplotlib.patches`

**Input Data:** `data/global-meat-production.csv`

**Output:** Bar chart showing continental meat production

---

### 6. **meat_supply_by_continent.py**
**Purpose:** Analyzes relationship between meat supply per capita and GDP per capita by continent.

**What it does:**
- Loads meat supply and GDP data by country
- Creates scatter plots with regional color coding
- Performs linear regression analysis
- Calculates correlation coefficients (R, R²)
- Provides both individual country and regional mean analysis
- Generates statistical summaries per region

**Dependencies:**
- `numpy`
- `pandas`
- `matplotlib`
- `matplotlib.patches`
- `matplotlib.lines`

**Input Data:** `data/Meat Supply Vs GDP Per capita.csv`

**Output:** 
- Scatter plot with regression line
- Regional mean analysis plot
- Console output with regression statistics

---

### 7. **meat_supply_vs_GDP_per_capita.py**
**Purpose:** Simplified version that plots meat supply vs GDP per capita with optional regional coloring.

**What it does:**
- Loads meat supply and GDP data
- Creates scatter plot with regional color differentiation
- Performs linear regression fitting
- Adds legend and annotations

**Dependencies:**
- `numpy`
- `pandas`
- `matplotlib`
- `matplotlib.patches`
- `matplotlib.lines`

**Input Data:** `data/Meat Supply Vs GDP Per capita.csv`

**Output:** Scatter plot with linear fit line

---

### 8. **InterestRateCalculator.py**
**Purpose:** Calculates compound interest growth over multiple years.

**What it does:**
- Takes initial savings, annual interest rate, and number of years
- Calculates compound interest year by year
- Uses the "Rule of 72" to estimate doubling time
- Prints year-by-year savings growth

**Dependencies:**
- None (uses built-in functions)

**Usage:** Run script (configured to run with $1000, 50% interest, 10 years)

---

### 9. **UnitTestingActivity.py**
**Purpose:** Demonstrates unit testing with a percentage change calculator.

**What it does:**
- Implements `calculate_change()` function to compute percentage change
- Includes 5 unit tests covering positive change, negative change, no change, large increases, large decreases
- Tests use `assertAlmostEqual()` for floating-point comparisons

**Dependencies:**
- `unittest` (built-in)

**Usage:** Run with `python -m unittest UnitTestingActivity.py`

---

### 10. **VersionControlActivity.py**
**Purpose:** Simple demonstration script for version control practice.

**What it does:**
- Prints three statements to demonstrate basic output

**Dependencies:**
- None (uses built-in functions)

---

### 11. **meat_production.py**
**Purpose:** Analyzes and visualizes meat production by country.

**What it does:**
- Loads meat production data by country
- Aggregates total production by entity
- Creates bar chart for top N countries (customizable)
- Displays production values on bars

**Dependencies:**
- `numpy`
- `pandas`
- `matplotlib`
- `matplotlib.patches`
- `matplotlib.lines`

**Input Data:** `data/meat-production-tonnes.csv`

**Output:** Bar chart showing top countries by meat production

---

## Required Data Files

Place the following CSV files in the `data/` directory:

- `HistoricalData_NVDA.csv` - NVIDIA stock price history
- `global-meat-production.csv` - Global meat production by continent
- `Meat Supply Vs GDP Per capita.csv` - Meat supply and GDP data by country
- `meat-production-tonnes.csv` - Meat production by country in tonnes

---

## Installation

### Install required packages:
```bash
pip install numpy pandas matplotlib
```

### Python version:
- Python 3.7 or higher

---

## File Structure
```
DAT5501_lab-2/
├── README.md
├── daily_price_change_sorting_activity.py
├── Duration_calculator.py
├── calendar_printer.py
├── advanced_data_fitting.py
├── continental_meat_production.py
├── meat_supply_by_continent.py
├── meat_supply_vs_GDP_per_capita.py
├── InterestRateCalculator.py
├── UnitTestingActivity.py
├── VersionControlActivity.py
├── meat_production.py
└── data/
    ├── HistoricalData_NVDA.csv
    ├── global-meat-production.csv
    ├── Meat Supply Vs GDP Per capita.csv
    └── meat-production-tonnes.csv
```