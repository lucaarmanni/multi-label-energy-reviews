# src/train.py

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
import config
from utils.text_cleaning import clean_text

# ============================================
# NUOVA CLASSE DBLossTrainer
# ============================================

class DBLossTrainer(Trainer):
    def __init__(self, *args, label_freq, bias_nu, num_classes, sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store pre-computed stats and move them to the correct device
        self.label_freq = torch.tensor(label_freq, dtype=torch.float).to(self.args.device)
        self.bias_nu = torch.tensor(bias_nu, dtype=torch.float).to(self.args.device)
        self.num_classes = num_classes
        self._sampler = sampler # Store the custom sampler

        # DB-Loss Hyperparameters (puoi spostarli in config.py se vuoi)
        self.alpha = 0.1
        self.beta = 10.0
        self.mu = 0.3
        self.lambda_ = 5.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(self.args.device)
        outputs = model(**inputs)
        logits = outputs.logits

        # --- 1. DYNAMIC WEIGHT CALCULATION (Re-balanced Weighting) ---
        p_c = (1 / self.num_classes) * (1 / self.label_freq)
        p_i_instance = (labels @ p_c.unsqueeze(1))
        raw_weights_r = p_c.unsqueeze(0) / (p_i_instance + 1e-9)
        smoothed_weights_r_hat = self.alpha + torch.sigmoid(self.beta * (raw_weights_r - self.mu))
        final_weights = smoothed_weights_r_hat
        
        # --- 2. NTR LOSS CALCULATION ---
        logits_shifted = logits - self.bias_nu.unsqueeze(0)
        loss_pos = F.binary_cross_entropy_with_logits(logits_shifted, labels, reduction="none") * (labels)
        logits_neg_scaled = self.lambda_ * logits_shifted
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg_scaled, labels, reduction="none") * (1 - labels) / self.lambda_

        loss = loss_pos + loss_neg
        weighted_loss = loss * final_weights
        final_loss = weighted_loss.mean()
        
        return (final_loss, outputs) if return_outputs else final_loss

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            sampler=self._sampler, 
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator, 
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )

# ============================================
# FUNZIONI UTILI (invariate)
# ============================================

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}

# ============================================
# LOGICA PRINCIPALE
# ============================================

def main():
    df = pd.read_excel(config.RAW_DATASET_PATH)
    print(f"Dataset loaded: {len(df)} samples")

    if config.APPLY_TEXT_CLEANING:
        df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].apply(clean_text)

    labels_df = df[config.LABEL_COLUMNS]
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(msss.split(df[config.TEXT_COLUMN], labels_df.values))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    # --- NUOVO: Pre-calcolo delle statistiche per DB-Loss ---
    print("\nPre-calcolo delle statistiche per la DB-Loss...")
    label_cols = config.LABEL_COLUMNS
    num_classes = len(label_cols)
    labels_np = train_df[label_cols].values
    label_freq = labels_np.sum(axis=0)
    kappa = 0.05
    class_priors = label_freq / len(train_df)
    bias_b = -np.log(1 / (class_priors + 1e-9) - 1)
    bias_nu = -kappa * bias_b
    print("Statistiche calcolate con successo.")

    # --- NUOVO: Creazione del Sampler (logica spostata qui) ---
    example_weights_val = []
    for row in labels_np:
        w = 0.0
        for i, v in enumerate(row):
            if v == 1:
                w += 1.0 / (label_freq[i] + 1e-6)
        example_weights_val.append(w if w > 0 else 1.0)
    
    w_tensor = torch.tensor(example_weights_val, dtype=torch.float)
    threshold = torch.quantile(w_tensor, 0.90)
    w_clamped = torch.clamp(w_tensor, max=threshold)
    sampler = WeightedRandomSampler(weights=w_clamped, num_samples=len(w_clamped), replacement=True)
    
    # --- Preparazione del Dataset (invariato) ---
    train_dataset = HFDataset.from_pandas(train_df)
    val_dataset = HFDataset.from_pandas(val_df)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)

    def preprocess_function(examples):
        # ... (questa funzione rimane uguale)
    
    # ... (tokenized_train e tokenized_val rimangono uguali)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_ID,
        num_labels=len(config.LABEL_COLUMNS),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=config.FINETUNED_MODEL_PATH,
        # ... (il resto degli argomenti rimane uguale)
    )

    # --- NUOVO: Inizializzazione del DBLossTrainer corretto ---
    trainer = DBLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(**config.EARLY_STOPPING_PARAMS)],
        label_freq=label_freq,
        bias_nu=bias_nu,
        num_classes=num_classes,
        sampler=sampler,
    )

    trainer.train()
    trainer.save_model(config.FINETUNED_MODEL_PATH)
    print("✅ Training completed successfully. Model saved!")

if __name__ == "__main__":
    main()
