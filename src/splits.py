import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset import TUMOR_COLS

VIEW_COLS = ['frontal', 'lateral', 'oblique']

# One fold for test, one for validation and five for training (about 71/14/14)
SPLIT_FOLDS = 7


def assign_lesion_groups(table: pd.DataFrame) -> pd.Series:
    table = table.sort_values('image_id').reset_index(drop=True)
    cols = [c for c in table.columns if c != 'image_id' and c not in VIEW_COLS]
    key = table[cols].astype(str).agg('|'.join, axis=1)
    group = (key != key.shift()).cumsum()
    return pd.Series(group.values, index=table['image_id'].values)


def stratify_labels(df: pd.DataFrame) -> np.ndarray:
    tier3 = df[TUMOR_COLS].values.argmax(axis=1)
    return np.where(df['tumor'].values == 1, tier3, len(TUMOR_COLS))


def group_split(table: pd.DataFrame, seed: int = 42) -> dict:
    table = table.assign(
        group=table['image_id'].map(assign_lesion_groups(table)).values
    )
    skf = StratifiedGroupKFold(
        n_splits=SPLIT_FOLDS, shuffle=True, random_state=seed
    )
    folds = [
        idx for _, idx in
        skf.split(table, stratify_labels(table), table['group'])
    ]
    parts = {
        'train': np.concatenate(folds[2:]),
        'val': folds[1],
        'test': folds[0],
    }
    return {
        name: table.iloc[idx].reset_index(drop=True)
        for name, idx in parts.items()
    }