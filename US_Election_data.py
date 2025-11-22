import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default file path
file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\US-2016-primary.csv"

def load_data(path=None):   
    # Load CSV using semicolon separator
    path = path if path else file_path
    return pd.read_csv(path, sep=';')

def _detect_columns(df):
    # Map lowercase column names to original names for easy matching
    cols = {c.lower(): c for c in df.columns}

    # Try to detect key fields automatically
    state_col = next((cols[k] for k in cols if "state" in k), None)
    fractionvote_col = next((cols[k] for k in cols if "fraction_votes" in k or "fractionvote" in k), None)
    cand_col  = next((cols[k] for k in cols if "candidate" in k or "name" in k), None)
    votes_col = next((cols[k] for k in cols if k in ("votes", "vote")), None)

    return state_col, cand_col, votes_col, fractionvote_col

def get_candidate_fractions(df, candidate_name):
    # Extract relevant columns (state, candidate, votes, fractions)
    state_col, cand_col, votes_col, frac_col = _detect_columns(df)

    # Must have at least state + candidate name
    if not state_col or not cand_col:
        raise ValueError("Required columns 'state' and 'candidate' not found in dataframe.")

    df = df.copy()

    # Case 1: dataset includes raw vote counts
    if votes_col:
        df[votes_col] = pd.to_numeric(df[votes_col], errors='coerce').fillna(0)

        # Sum total votes per state
        total_by_state = df.groupby(state_col)[votes_col].sum()

        # Normalize candidate name and search for matches
        cand_series = df[cand_col].astype(str)
        mask = cand_series.str.lower() == candidate_name.lower()

        # If exact match fails, try substring match
        if not mask.any():
            mask = cand_series.str.lower().str.contains(candidate_name.lower(), na=False)
            if not mask.any():
                raise ValueError(f"Candidate '{candidate_name}' not found in '{cand_col}' column.")

        # Sum candidate votes per state
        cand_by_state = df[mask].groupby(state_col)[votes_col].sum()

        # Compute fraction = candidate votes / total votes
        frac = (cand_by_state / total_by_state).reindex(total_by_state.index).fillna(0)
        frac.name = candidate_name
        return frac

    # Case 2: dataset includes precomputed vote fractions
    if frac_col:
        df[frac_col] = pd.to_numeric(df[frac_col], errors='coerce').fillna(0)

        # Find rows corresponding to candidate
        cand_series = df[cand_col].astype(str)
        mask = cand_series.str.lower() == candidate_name.lower()

        # Retry substring match if exact match fails
        if not mask.any():
            mask = cand_series.str.lower().str.contains(candidate_name.lower(), na=False)
            if not mask.any():
                raise ValueError(f"Candidate '{candidate_name}' not found in '{cand_col}' column.")

        # If vote column exists, compute weighted mean of fractions
        if votes_col:
            df[votes_col] = pd.to_numeric(df[votes_col], errors='coerce').fillna(0)

            def wavg(g):
                w = g[votes_col].astype(float)
                if w.sum() == 0:
                    # Fall back to simple mean if no vote data
                    return g[frac_col].astype(float).mean()
                return np.average(g[frac_col].astype(float), weights=w)

            # Weighted mean fraction per state
            weighted = df[mask].groupby(state_col).apply(wavg)
            all_states = pd.Index(df[state_col].unique())
            frac = weighted.reindex(all_states).fillna(0)
            frac.name = candidate_name
            return frac

        # Otherwise fallback to simple mean fraction per state
        mean_frac = df[mask].groupby(state_col)[frac_col].mean()
        all_states = pd.Index(df[state_col].unique())
        frac = mean_frac.reindex(all_states).fillna(0)
        frac.name = candidate_name
        return frac

def plot_candidate_histogram(df, candidate_name, bins=20, figsize=(9,6), save_path=None, **plt_kwargs):
    # Compute candidate's vote fraction per state
    frac = get_candidate_fractions(df, candidate_name)
    frac = frac.sort_index()

    # Create histogram plot
    plt.figure(figsize=figsize)
    plt.hist(frac.values, bins=bins, range=(0,1),
             color='C0', edgecolor='k', **plt_kwargs)

    # Label axes and title
    plt.xlabel(f"Fraction of votes for '{candidate_name}' (per state)")
    plt.ylabel("Number of states / districts")
    plt.title(f"Histogram of per-state vote fraction for {candidate_name}")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Optionally save file
    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()
    return frac

if __name__ == "__main__":
    # Load file and show histogram for Donald Trump
    df = load_data()
    frac_series = plot_candidate_histogram(df, "Donald Trump", bins=15)
    print(frac_series.sort_index())
