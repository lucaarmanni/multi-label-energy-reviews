[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucaarmanni/multi-label-energy-reviews/blob/main/demo_inference.ipynb)

# Multi-Label Classification of Customer Reviews in the Energy Sector

This repository contains the implementation and fine-tuning code for a **multi-label text classification model** applied to Italian customer reviews in the **energy sector**.

The model is based on **Musixmatch/umberto-commoncrawl-cased-v1**, further adapted through **Domain-Adaptive Pretraining (DAPT)** and fine-tuned on a manually labeled dataset of real customer reviews.


---

## 🎯 Objective

Automatically assign one or more thematic labels to customer reviews of energy providers, enabling large-scale feedback analysis and quantitative insights into customer experience.

---

## 🧠 Model Overview

- **Base model:** Musixmatch/umberto-commoncrawl-cased-v1  
- **Domain adaptation:** DAPT on ~34k unlabeled energy reviews  
- **Fine-tuning objective:** Multi-label classification  
- **Loss function:** Distribution-Balanced Loss (β=0.95, γ=2.5, λ=2.0)  
- **Sampler:** WeightedRandomSampler  
- **Framework:** PyTorch + Hugging Face Transformers  
- **Tracking:** Weights & Biases (wandb)



## 🧾 Dataset

| Dataset | Description | Samples |
|----------|--------------|----------|
| [`training_dataset.xlsx`](./data/training_dataset.xlsx) | Manually labeled dataset | 1,742 |
| [`training_dataset_augmented.xlsx`](./data/training_dataset_augmented.xlsx) | Back-translated dataset (+335 synthetic reviews) | 2,077 |
| [`sample_input.xlsx`](./data/sample_input.xlsx) | sample dataset for testing | 10  | quick test |
| [`Recensioni_OCTOPUS.xlsx`](./data/Recensioni_OCTOPUS.xlsx) | Real-world dataset of customer reviews for octopus energy | 10279  | Full Inference |
Each review can have multiple labels (multi-hot encoding).  
Data were preprocessed with a text cleaning function preserving casing and punctuation.



## 📊 Model Performance

The fine-tuned **BT-umBERTo-DAPT** model achieved strong results on the validation set using **Macro F1** as the primary metric.  

Full details are provided in the [final report](https://github.com/lucaarmanni/multi-label-energy-reviews/blob/main/reports/report.pdf).







