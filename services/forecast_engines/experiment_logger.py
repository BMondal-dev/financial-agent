import json
import os
from datetime import datetime

EXPERIMENT_PATH = "../data/experiments.json"


def log_experiment(target, neighbors, mae, predicted_return):

    if not os.path.exists(EXPERIMENT_PATH):
        with open(EXPERIMENT_PATH, "w") as f:
            json.dump([], f)

    with open(EXPERIMENT_PATH, "r") as f:
        data = json.load(f)

    data.append({
        "target": target,
        "neighbors": neighbors,
        "mae": mae,
        "predicted_return": predicted_return,
        "timestamp": datetime.utcnow().isoformat()
    })

    with open(EXPERIMENT_PATH, "w") as f:
        json.dump(data, f, indent=2)