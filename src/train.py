# src/train.py

# ============================================
# Fine-tuning script (for reference only)
# ============================================

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
import config
from utils.text_cleaning import clean_text


class DistributionBalancedLoss(nn.Module):
    def __init__(self, label_freq, beta=0.95, gamma=2.0, lambda_=2.0):
        super().__init__()
        effective_num = 1 - torch.pow(beta, label_freq)
        weights = (1 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * label_freq.numel()
        self.register_buffer("class_weight", weights)
        self.gamma = gamma
        self.lambda_ = lambda_

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        cb = bce * self.class_weight.unsqueeze(0)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t).pow(self.gamma) * bce
        loss = (cb + self.lambda_ * focal) / (1 + self.lambda_)
        return loss.mean()


class DBLossTrainer(Trainer):
    def __init__(self, *args, label_freq, **kwargs):
        super().__init__(*args, **kwargs)
        freq = torch.tensor(label_freq, dtype=torch.float, device=self.args.device)
        self.db_loss = DistributionBalancedLoss(
            freq,
            beta=config.LOSS_PARAMS["beta"],
            gamma=config.LOSS_PARAMS["gamma"],
            lambda_=config.LOSS_PARAMS["lambda_"],
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.db_loss(outputs.logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


def main():
    df = pd.read_excel(config.RAW_DATASET_PATH)
    print(f"Dataset loaded: {len(df)} samples")

    if config.APPLY_TEXT_CLEANING:
        df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].apply(clean_text)

    labels_df = df[config.LABEL_COLUMNS]
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(msss.split(df[config.TEXT_COLUMN], labels_df.values))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    label_freq = train_df[config.LABEL_COLUMNS].sum().values

    train_dataset = HFDataset.from_pandas(train_df)
    val_dataset = HFDataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples[config.TEXT_COLUMN],
            truncation=True,
            max_length=config.TRAINING_PARAMS["max_length"],
        )
        tokenized["labels"] = [
            [float(label) for label in labels]
            for labels in zip(*[examples[col] for col in config.LABEL_COLUMNS])
        ]
        return tokenized

    tokenized_train = train_dataset.map(
        preprocess_function, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        preprocess_function, batched=True, remove_columns=val_dataset.column_names
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_ID,
        num_labels=len(config.LABEL_COLUMNS),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=config.FINETUNED_MODEL_PATH,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        **config.TRAINING_PARAMS,
    )

    trainer = DBLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(**config.EARLY_STOPPING_PARAMS)],
        label_freq=label_freq,
    )

    trainer.train()
    trainer.save_model(config.FINETUNED_MODEL_PATH)
    print("✅ Training completed successfully. Model saved!")


if __name__ == "__main__":
    main()
