import fcntl
import json
import os
from datetime import datetime, timezone

EXPERIMENT_PATH = "../data/experiments.json"


def log_experiment(
    target: str,
    neighbors: list[str],
    horizon: int,
    mae: float,
    predicted_return: float,
    *,
    mae_baseline_zero: float | None = None,
    mae_baseline_mean: float | None = None,
    beats_baseline_zero: bool | None = None,
    mae_for_ranking: float | None = None,
    cv_mae_mean: float | None = None,
    cv_mae_std: float | None = None,
    cv_worst_split_mae: float | None = None,
    cv_worst_split_test_date_start: str | None = None,
    cv_worst_split_test_date_end: str | None = None,
    train_rows: int | None = None,
    test_rows: int | None = None,
    train_date_start: str | None = None,
    train_date_end: str | None = None,
    test_date_start: str | None = None,
    test_date_end: str | None = None,
    experiment_id: str | None = None,
    rationale: str | None = None,
    source: str | None = None,
    orchestrator_round: int | None = None,
):
    canonical_neighbors = sorted(neighbors)
    entry = {
        "target": target,
        "neighbors": canonical_neighbors,
        "horizon": horizon,
        "mae": mae,
        "predicted_return": predicted_return,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry["mae_baseline_zero"] = mae_baseline_zero
    entry["mae_baseline_mean"] = mae_baseline_mean
    entry["beats_baseline_zero"] = beats_baseline_zero
    entry["mae_for_ranking"] = mae_for_ranking
    entry["cv_mae_mean"] = cv_mae_mean
    entry["cv_mae_std"] = cv_mae_std
    entry["cv_worst_split_mae"] = cv_worst_split_mae
    entry["cv_worst_split_test_date_start"] = cv_worst_split_test_date_start
    entry["cv_worst_split_test_date_end"] = cv_worst_split_test_date_end
    entry["train_rows"] = train_rows
    entry["test_rows"] = test_rows
    entry["train_date_start"] = train_date_start
    entry["train_date_end"] = train_date_end
    entry["test_date_start"] = test_date_start
    entry["test_date_end"] = test_date_end
    entry["experiment_id"] = experiment_id
    entry["rationale"] = rationale
    entry["source"] = source
    entry["orchestrator_round"] = orchestrator_round

    abs_path = os.path.abspath(EXPERIMENT_PATH)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    lock_path = abs_path + ".lock"

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(abs_path):
                with open(abs_path) as f:
                    data = json.load(f)
            else:
                data = []
            data.append(entry)
            with open(abs_path, "w") as f:
                json.dump(data, f, indent=2)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
