# DisasterSense AI

**An experimental machine learning prototype for predicting short-term river water-level rises.**

## About

DisasterSense AI studies river water-level patterns at Champua station on the Baitarani River in Kendujhar, Odisha, India.

It uses historical water-level observations and NASA POWER rainfall data to train and compare two Random Forest classification models.

### Models
1. Water-only model
2. Rainfall-enhanced model

The prediction target is a water-level rise of at least 0.10 metres over the following three hours. This is a project-defined experimental threshold, not an official flood-warning threshold.

## Model evaluation

Evaluation used a chronological data split with gaps between partitions. The test period was inspected during development, so these results should be treated as a robustness check rather than a pristine independent benchmark.

| Metric | Water-only | Rainfall-enhanced |
|---|---:|---:|
| Precision | 0.500 | 0.597 |
| Recall | 0.511 | 0.394 |
| F1-score | 0.505 | 0.474 |
| Detected events | 48 | 37 |
| Missed events | 46 | 57 |
| False alarms | 48 | 25 |

The rainfall-enhanced model produced fewer false alarms but missed more events. The water-only model achieved higher recall and F1-score in this evaluation.

## Technology

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- NASA POWER rainfall data
- Central Water Commission (CWC) water-level observations

## Run locally

1. Install Python.
2. Clone or download this repository.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Run:

   `streamlit run app.py`

Keep the model files in the same folder as `app.py`.

Historical analysis requires the corresponding CSV datasets. These are not included in the public repository by default.

## Project limitations and safety

**DisasterSense AI is an experimental research prototype, not an operational flood-warning system.**

It has not been validated for real-world flood forecasting, public alerts, evacuation decisions, or emergency response. Predictions may be incorrect or miss important events. Always follow official CWC, IMD, and government disaster-management advisories.

The 0.10 m threshold is defined for this project and is not an official warning threshold.

## Data and attribution

Water-level observations are associated with the Central Water Commission (CWC). Rainfall features were derived from NASA POWER data.

Check the relevant data-use and redistribution terms before publishing source datasets or derived datasets. Cite the original providers when using their data.

## Dashboard Preview

### Main Dashboard
![Main Dashboard](assets/Screenshot2026-10-31154112.png)

### Model Predictions
![Model Predictions](assets/Screenshot2026-10-31154130.png)

### Model Performance
![Model Performance](assets/Screenshot2026-10-31154144.png)

### Historical Analysis
![Historical Analysis](assets/Screenshot2026-10-31154216.png)

## Author

Krish Deep Dhanjal
