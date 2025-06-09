import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

def main():
    # Read the single input line
    parts = sys.stdin.read().strip().split()
    if len(parts) < 2:
        return
    portfolio_name = parts[0]
    portfolio_pnl = np.array(list(map(float, parts[1:])))

    # Load the data
    returns_df = pd.read_csv('stocks_returns.csv')
    metadata_df = pd.read_csv('stocks_metadata.csv')

    # Identify the ID and cost columns in metadata_df
    id_cols = [c for c in metadata_df.columns if 'id' in c.lower()]
    cost_cols = [c for c in metadata_df.columns if 'cost' in c.lower()]
    if not id_cols or not cost_cols:
        raise KeyError("Missing 'id' or 'cost' column in metadata")
    id_col = id_cols[0]
    cost_col = cost_cols[0]

    # Build list of stock IDs and align capital costs
    stock_ids = [c for c in returns_df.columns if c != 'date']
    capital_costs = (
        metadata_df
        .set_index(id_col)[cost_col]
        .reindex(stock_ids)
        .fillna(1.0)
        .values
    )

    # Prepare regression matrix X and target y
    returns = returns_df[stock_ids].values / 100.0
    X = returns / capital_costs
    y = portfolio_pnl

    # Fit Lasso with cross-validation
    model = LassoCV(
        cv=5,
        max_iter=10000,
        alphas=np.logspace(-4, 0, 50),
        random_state=42
    )
    model.fit(X, y)

    # Compute hedge quantities: q_i = -beta_i / cost_i
    beta = model.coef_
    quantities = -beta / capital_costs

    # Print non-zero hedges
    for sid, q in zip(stock_ids, quantities):
        q_int = int(round(q))
        if q_int != 0:
            print(f"{sid} {q_int}")

if __name__ == "__main__":
    main()
