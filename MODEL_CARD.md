# Model Card: Classificatore Multi-Label di Recensioni (Settore Energia)

**Versione:** 1.0
**Sviluppato da:** Luca Armanni
**Data:**  Ottobre 2025
**Architettura del Modello:** Modello basato su **umBERTo**, sottoposto a **Domain-Adaptive Pre-Training (DAPT)** su 35.000 recensioni del settore energetico, e successivamente **fine-tunato** per un task di classificazione multi-etichetta.

---

## 1. Scopo del Modello (Intended Use)

* **Utilizzo Previsto:** Questo modello è stato addestrato per analizzare recensioni di clienti in lingua italiana, specifiche del settore energetico, e assegnare una o più etichette tematiche per l'analisi aggregata del feedback.
* **Insieme di Etichette Possibili:** `Label-1`, `Label-3`, `Label-5`, `Label-6`, `Label-7`, `Label-9`, `Label-10`, `Label-12`, `Label-13`, `Label-14`, `Label-16`, `Label-17`, `Label-19`, `Label-21`, `Label-22`, `Label-26`, `Label-32`, `Label-34`, `Label-35`, `Label-38`, `Label-42`.
* **Utenti Previsti:** Team di Customer Experience, analisti di dati, product manager del settore energetico.
* **Utilizzi Fuori Scopo (Out-of-Scope):** Il modello non è progettato per prendere decisioni automatizzate su singoli clienti o per operare al di fuori del dominio energetico italiano.

---

## 2. Dati (Data)

### Dati di Addestramento
* **Fonte:** Dataset personalizzato (`training_dataset.xlsx`) di recensioni di clienti, etichettato manualmente.
* **Strategie:** Per affrontare lo sbilanciamento delle etichette e la scarsità di dati, sono state usate due tecniche:
    1.  **Data Augmentation:** Il training set è stato ampliato utilizzando la **back-translation**.
    2.  **Data Sampling:** È stato utilizzato un `WeightedRandomSampler` per aumentare la frequenza di campionamento delle classi meno rappresentate durante il training.

### Dati di Valutazione
* **Fonte:** Il set di validazione (20% del totale) è composto da **350 campioni originali non aumentati**, ottenuti tramite uno split stratificato (`MultilabelStratifiedShuffleSplit`).

---

## 3. Algoritmo di Addestramento (Training Algorithm)

* **Algoritmo:** Fine-tuning dell'architettura `umBERTo` post-DAPT.
* **Funzione di Perdita (Loss Function):** **`DistributionBalancedLoss`** (`beta=0.95`, `gamma=2.5`, `lambda=2.0`).
* **Parametri Chiave:** `learning_rate=3e-5`, `weight_decay=0.05`, Early Stopping (`patience=5`).

---
## 4. Performance del Modello (Model Performance)

I seguenti risultati sono stati ottenuti sul set di validazione **senza ottimizzazione delle soglie di decisione (threshold tuning)**, utilizzando una soglia fissa pari a 0.5 per tutte le etichette.

| Metrica          | Punteggio Finale | Descrizione                                                                                                                                     |
| :--------------- | :--------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Macro F1-Score** | **0.524** | Media delle F1-Score calcolate per ogni singola etichetta. Questa metrica penalizza le classi meno rappresentate e misura la capacità del modello di gestire la distribuzione sbilanciata. |
| **Micro F1-Score** | **0.771** | F1 calcolata globalmente su tutte le predizioni, fornendo una misura aggregata della performance complessiva del modello. |

---

### Nota sulla versione del modello

Questa versione del modello (**BT-umBERTo-DAPT**) corrisponde alla release pubblica fornita nella repository, destinata a un utilizzo pratico per l'inferenza automatica.  
Le versioni successive, utilizzate durante la sperimentazione interna, hanno incluso un’ottimizzazione delle soglie per etichetta (threshold tuning) e hanno raggiunto **Macro F1 = 0.615** e **Micro F1 = 0.783**.  
Tuttavia, tali configurazioni non sono fornite in questa release per privilegiare semplicità e riproducibilità.

---

## 5. Limiti e Bias (Limitations and Bias)

* **Limiti Tecnici:** Il modello è altamente specializzato sul dominio energetico italiano e non generalizza ad altri contesti. Le performance sulle etichette meno frequenti, sebbene migliorate, rimangono inferiori rispetto a quelle più comuni.

* **Bias:** Il modello riflette la distribuzione tematica e linguistica del corpus di training. Eventuali bias presenti nei dati di origine saranno appresi e replicati dal modello.
