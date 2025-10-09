[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucaarmanni/multi-label-energy-reviews/blob/main/demo_inference.ipynb)
# ⚡ Multi-Label Classification of Energy Sector Reviews with BT-umBERTo-DAPT

Questo progetto implementa un sistema di **multi-label text classification** per recensioni di clienti nel settore energetico,  
basato su **BT-umBERTo-DAPT**, una versione di *Musixmatch/umBERTo-commoncrawl-cased-v1* adattata al dominio energetico (Domain-Adaptive Pretraining).

Il modello è stato fine-tunato per riconoscere più categorie contemporaneamente (es. *Servizio clienti, Tariffe, pagamenti *, ecc.)  
ed è pronto per l’inferenza su qualsiasi dataset di recensioni in lingua italiana.

---

## 🧠 Modello Fine-Tunato

- Nome modello: `BT-aug-umBERTo-finetunedB-classifier:v0`  
- Base model: `Musixmatch/umBERTo-commoncrawl-cased-v1`  
- Task: Multi-Label Text Classification 
- Framework: PyTorch + Hugging Face Transformers  
- Download:  
  👉 [**Scarica il modello da Google Drive**](https://drive.google.com/file/d/1gXgNqfRy89gfFZkSVsdyLaxspD60ZPUF/view?usp=drive_link)

---

## ⚙️ Esecuzione su Google Colab

Per testare il modello su **Google Colab** (consigliata GPU T4):

```python
# 🔧 1️⃣ Clona la repository e installa le dipendenze
!git clone https://github.com/lucaarmanni/multi-label-energy-reviews.git
%cd multi-label-energy-reviews/src
!pip install -r ../requirements.txt

# 💾 2️⃣ Monta Google Drive ed estrai il modello
from google.colab import drive
import zipfile, os

drive.mount('/content/drive')

drive_path = "/content/drive/MyDrive/classifier_multi_label_energy_reviews/BT-aug-umBERTo-finetunedB-classifier:v0.zip"
destination_path = "/content/multi-label-energy-reviews/models/"

os.makedirs(destination_path, exist_ok=True)

with zipfile.ZipFile(drive_path, 'r') as zip_ref:
    zip_ref.extractall(destination_path)

print("✅ Modello estratto correttamente in:", destination_path)

# 🚀 3️⃣ Avvia l'inferenza
%cd /content/multi-label-energy-reviews/src
!python inference.py

