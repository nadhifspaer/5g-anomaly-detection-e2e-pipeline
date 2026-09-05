# Per-feature deviation-contribution ranking against a row's own (cell_id,
# slice_type) rolling baseline. See architecture.md Section 3, Section 5 Step 5.

import pandas as pd

from src import feature_engineering as fe


def rank_deviation_contributions(X: pd.DataFrame, row_index, top_n: int = 5) -> pd.DataFrame:
    """Ranks this row's own (cell_id, slice_type) KPI deviations by |z-score|; no SHAP, no other black-box explainer, no cross-cell comparison."""
    row = X.loc[row_index]
    kpi_z = row[fe.ZSCORE_COLS].astype(float)
    order = kpi_z.abs().sort_values(ascending=False).index
    ranked = kpi_z.loc[order].head(top_n)
    return pd.DataFrame(
        {
            "kpi": [col.removesuffix("_zscore") for col in ranked.index],
            "zscore": ranked.to_numpy(),
        }
    )
