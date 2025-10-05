
# =========================================================
# PATHS (valid if executed from inside src/)
# =========================================================
RAW_DATASET_PATH = "../data/training_dataset_augmented.xlsx"
FINETUNED_MODEL_PATH = "../models/final-umBERTo-DAPT"

# =========================================================
# DATASET CONFIGURATION
# =========================================================
TEXT_COLUMN = "Reviewtext"
LABEL_COLUMNS = [f"Label-{i}" for i in range(1, 22)]
APPLY_TEXT_CLEANING = False

# =========================================================
# MODEL CONFIGURATION
# =========================================================
BASE_MODEL_ID = "Musixmatch/umberto-commoncrawl-cased-v1"

# =========================================================
# TRAINING PARAMETERS
# (keep in case you later retrain)
# =========================================================
TRAINING_PARAMS = {
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.05,
    "gradient_accumulation_steps": 2,
    "logging_steps": 100,
    "save_steps": 100,
    "eval_steps": 100,
    "max_length": 128,
    "seed": 42,
}

EARLY_STOPPING_PARAMS = {
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0,
}

LOSS_PARAMS = {"beta": 0.95, "gamma": 2.5, "lambda_": 2.0}
