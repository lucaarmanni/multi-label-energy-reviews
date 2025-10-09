# src/config.py

# =========================================================
# PATHS
# =========================================================
RAW_DATASET_PATH = "../data/training_dataset_augmented.xlsx"
# CONSIGLIO: Cambia il nome per non sovrascrivere il vecchio modello
FINETUNED_MODEL_PATH = "../models/final-umBERTo-DBLoss" 

# =========================================================
# DATASET CONFIGURATION
# =========================================================
TEXT_COLUMN = "Reviewtext"
LABEL_COLUMNS = [f"Label-{i}" for i in range(1, 22)]
APPLY_TEXT_CLEANING = False # Lascia a False se hai già pulito il testo

# =========================================================
# MODEL CONFIGURATION
# =========================================================
BASE_MODEL_ID = "Musixmatch/umberto-commoncrawl-cased-v1"

# =========================================================
# TRAINING PARAMETERS
# =========================================================
TRAINING_PARAMS = {
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 20, # Aumentato per dare più tempo al modello
    "warmup_ratio": 0.1,
    "weight_decay": 0.05,
    "gradient_accumulation_steps": 2,
    "logging_steps": 50, # Abbassato per avere log più frequenti
    "save_steps": 50,
    "eval_steps": 50,
    "max_length": 128,
    "seed": 42,
}

EARLY_STOPPING_PARAMS = {
    "early_stopping_patience": 4, # Aumentato leggermente
    "early_stopping_threshold": 0.0,
}

# RIMOSSO: La vecchia sezione LOSS_PARAMS non è più necessaria
# I nuovi iperparametri (alpha, beta, mu, lambda_) sono dentro DBLossTrainer
