# ============================================
# inference.py 
# ============================================


import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.text_cleaning import clean_text

# =========================================================
# CONFIGURAZIONE
# =========================================================
# ID del tuo modello su Hugging Face Hub
MODEL_ID = "lucaarmanni/umberto-energy-reviews-classifier" 

# Percorso dei dataset di input e output
DATA_DIR = os.path.abspath("../data")
DATASETS = [
    ("sample_input.xlsx", "predicted_sample_input.xlsx"),
    ("Recensioni_OCTOPUS.xlsx", "predicted_octopus_reviews.xlsx"),
]

# =========================================================
# CARICAMENTO MODELLO DA HUB
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 Avvio inferenza... (device: {device})")
print(f"Caricamento modello da Hugging Face Hub: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
model.eval()

label_mapping = {int(k): v for k, v in model.config.id2label.items()}
print(f"   ✅ Trovate {len(label_mapping)} etichette nella configurazione del modello.\n")

# =========================================================
# FUNZIONE DI INFERENZA
# =========================================================
def run_inference(input_filename, output_filename):
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"⚠️ File non trovato: {input_path} — saltato.\n")
        return

    print(f"\n📂 Esecuzione inferenza su: {input_filename}")
    df = pd.read_excel(input_path)
    
    # Assumi che la colonna con il testo si chiami 'Reviewtext' o simile
    text_col = "Reviewtext" 
    if text_col not in df.columns:
        raise ValueError(f"❌ Il file {input_filename} deve contenere una colonna chiamata '{text_col}'.")

    df["CleanedText"] = df[text_col].apply(clean_text)
    predicted_labels = []

    for text in tqdm(df["CleanedText"], desc=f"Predicting on {input_filename}", total=len(df)):
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            labels = [label_mapping[i] for i, p in enumerate(probs) if p >= 0.5]
        predicted_labels.append(", ".join(labels) if labels else "None")

    df["Predicted_Labels"] = predicted_labels
    df.drop(columns=["CleanedText"], inplace=True)
    df.to_excel(output_path, index=False)

    print(f"✅ File salvato in: {output_path}")
    print("--- Anteprima dei risultati ---")
    print(df[[text_col, "Predicted_Labels"]].head())
    print("--------------------------------\n")

# =========================================================
# ESECUZIONE
# =========================================================
if __name__ == "__main__":
    for input_name, output_name in DATASETS:
        run_inference(input_name, output_name)
    print("🏁 Tutte le inferenze completate con successo!")
