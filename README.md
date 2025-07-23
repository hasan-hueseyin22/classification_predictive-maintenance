# Predictive Maintenance Classification

## 📝 Projektbeschreibung

Dieses Projekt zielt darauf ab, Maschinenausfälle in einem industriellen Umfeld vorherzusagen. Anhand von Sensordaten wie Lufttemperatur, Prozesstemperatur, Rotationsgeschwindigkeit, Drehmoment und Werkzeugverschleiß wird ein Klassifikationsmodell trainiert, das prognostiziert, ob eine Maschine ausfallen wird oder nicht. 

Das Ziel ist die Reduzierung von ungeplanten Ausfallzeiten und die Optimierung von Wartungsplänen, was zu erheblichen Kosteneinsparungen führen kann.

**Datensatz:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) von der UCI Machine Learning Repository.

## 🛠️ Tech Stack

- **Python**
- **Pandas** für die Datenmanipulation
- **Scikit-learn** für das Machine Learning Modell (Random Forest Classifier) und die Vorverarbeitung
- **Joblib** zum Speichern des trainierten Modells

## 🚀 Installation und Ausführung

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/hasan-hueseyin22/classification_predictive-maintenance.git](https://github.com/hasan-hueseyin22/predictive-maintenance-classification.git)
    cd predictive-maintenance-classification
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren (empfohlen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Modell trainieren:**
    Das Skript lädt die Daten automatisch herunter, verarbeitet sie vor und trainiert das Modell.
    ```bash
    python src/train.py
    ```
## 📊 Ergebnisse

Nach dem Training gibt das Skript einen Klassifikationsbericht aus, der Metriken wie Präzision, Recall und F1-Score für jede Klasse (Ausfall/kein Ausfall) enthält. Das trainierte Modell wird als `models/model.joblib` gespeichert und kann für zukünftige Vorhersagen verwendet werden.
## 📂 Repository-Struktur
```
predictive-maintenance-classification/
├── data/               # Datensätze (roh und verarbeitet)
├── models/             # Gespeicherte Modelle
├── notebooks/          # Jupyter Notebooks für die explorative Datenanalyse
├── src/                # Quellcode
│   ├── config.py       # Konfigurationsdatei
│   ├── data_preprocessing.py # Skripte zur Datenvorverarbeitung
│   ├── model.py        # Modelldefinition
│   └── train.py        # Trainingsskript
├── .gitignore          # Git-Ignore-Datei
├── README.md           # Diese Datei
└── requirements.txt    # Python-Abhängigkeiten
```
