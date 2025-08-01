import json
from datetime import datetime

RETRAIN_LOG = "retrain_history.json"

def log_retrain_history(history, model_version):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_version": model_version,
        "train_accuracy": history.history['accuracy'][-1],
        "val_accuracy": history.history['val_accuracy'][-1]
    }
    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(RETRAIN_LOG, "w") as f:
        json.dump(logs, f, indent=2)

def get_next_model_version():
    existing = [f for f in os.listdir("models") if f.startswith("model_v") and f.endswith(".h5")]
    versions = [int(f.split("_v")[1].split(".")[0]) for f in existing if "_v" in f]
    return max(versions) + 1 if versions else 1

def save_model(model, model_version):
    model.save(f"models/model_v{model_version}.h5")