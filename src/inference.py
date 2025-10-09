# ============================================
# inference.py 
# ============================================

import os
import torch
import zipfile
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.text_cleaning import clean_text

# =========================================================
# CONFIGURAZIONE
# =========================================================

# Percorso della cartella modelli nella repo
BASE_MODEL_DIR = os.path.abspath("../models")
MODEL_FOLDER = os.path.join(BASE_MODEL_DIR, "best-umBERTo-DBLoss-BT")
MODEL_ZIP = os.path.join(BASE_MODEL_DIR, "best-umBERTo-DBLoss-BT.zip")

# Percorso dei dataset
DATA_DIR = os.path.abspath("../data")
DATASETS = [
    ("sample_input.xlsx", "predicted_sample_input.xlsx"),
    ("Recensioni_OCTOPUS.xlsx", "predicted_octopus_reviews.xlsx"),
]

# =========================================================
# 1️⃣ Caricamento o estrazione automatica del modello
# =========================================================
if not os.path.exists(MODEL_FOLDER):
    if os.path.exists(MODEL_ZIP):
        print(f"📦 File zip del modello trovato: {MODEL_ZIP}")
        print("Estrazione in corso...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(BASE_MODEL_DIR)
        print("✅ Modello estratto correttamente in:", MODEL_FOLDER)
    else:
        print("❌ ERRORE: il modello non è presente nella cartella 'models/'.")
        print("Scarica il file .zip dal link indicato in models/README.md")
        raise FileNotFoundError("Modello mancante. Consulta il file models/README.md per scaricarlo.")

# =========================================================
# 2️⃣ Caricamento del tokenizer e del modello
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 Avvio inferenza... (device: {device})")
print(f"Caricamento modello da: {MODEL_FOLDER}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER, local_files_only=True).to(device)
model.eval()

# =========================================================
# 3️⃣ Ricostruzione dinamica della mappatura etichette
# =========================================================
TRAINING_FILE = os.path.join(DATA_DIR, "training_dataset.xlsx")
try:
    print(f"\n🔍 Lettura della mappatura etichette da: {TRAINING_FILE}")
    df_train = pd.read_excel(TRAINING_FILE)
    label_cols = [col for col in df_train.columns if col.startswith("Label-")]
    if not label_cols:
        raise ValueError("Nessuna colonna trovata con prefisso 'Label-'.")
    label_mapping = {i: name for i, name in enumerate(label_cols)}
    print(f"   ✅ Trovate {len(label_cols)} etichette nel dataset di training.\n")
except Exception as e:
    print(f"⚠️ Impossibile ricostruire la mappatura etichette: {e}")
    label_mapping = {i: f"Label-{i+1}" for i in range(model.config.num_labels)}

# =========================================================
# 4️⃣ Funzione per eseguire inferenza su un dataset
# =========================================================
def run_inference(input_filename, output_filename):
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"⚠️ File non trovato: {input_path} — saltato.\n")
        return

    print(f"\n📂 Esecuzione inferenza su: {input_filename}")
    df = pd.read_excel(input_path)

    if "Reviewtext" not in df.columns:
        raise ValueError(f"❌ Il file {input_filename} deve contenere una colonna chiamata 'Reviewtext'.")

    df["CleanedText"] = df["Reviewtext"].apply(clean_text)
    predicted_labels = []

    for text in tqdm(df["CleanedText"], desc="Predicting", total=len(df)):
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            labels = [label_mapping[i] for i, p in enumerate(probs) if p >= 0.5]
        predicted_labels.append(", ".join(labels) if labels else "None")

    df["Predicted_Labels"] = predicted_labels
    df.drop(columns=["CleanedText"], inplace=True)
    df.to_excel(output_path, index=False)

    print(f"✅ File salvato in: {output_path}")
    print(df[["Reviewtext", "Predicted_Labels"]].head())
    print("--------------------------------------------------\n")

# =========================================================
# 5️⃣ Esegui inferenza su tutti i dataset disponibili
# =========================================================
for input_name, output_name in DATASETS:
    run_inference(input_name, output_name)

print("🏁 Tutte le inferenze completate con successo!")


