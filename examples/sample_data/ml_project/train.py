"""Tiny sample ML project so `carto` works out of the box. Replace examples/ with your own folders."""
import logging

logger = logging.getLogger(__name__)


def train_model(X, y, seed: int = 0) -> dict:
    """Fit a baseline model with a held-out split and report calibrated metrics."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=seed)
        model = LogisticRegression().fit(Xtr, ytr)
        return {"accuracy": model.score(Xte, yte)}
    except Exception as e:
        logger.error("training failed: %s", e)
        raise
