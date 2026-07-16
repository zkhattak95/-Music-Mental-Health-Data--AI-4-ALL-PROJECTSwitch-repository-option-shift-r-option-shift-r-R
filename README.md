# Mind & Melody

Mind & Melody is an AI4ALL Group 11C capstone project that explores how student lifestyle patterns may relate to burnout risk and how music-listening patterns may support personal wellness.

The application combines:

- A student burnout-risk check-in
- A Random Forest classification pipeline
- Data-informed music suggestions
- Interactive burnout and music-survey visualizations
- Responsible-use guidance and clear model limitations

> **Important:** This is an educational wellness prototype. It does not diagnose burnout or any mental-health condition and is not a substitute for professional support.

## Team

- Nelson Kwesi Xedzro
- Estela Acosta Herrera
- Zeeshan Khattak
- Oyagbemi Oluwatoba
- Raheem Kadiku
- Sahasra Kothembaka

## Application pages

1. **Home** - Introduces the project and its purpose.
2. **Burnout Check-In** - Collects lifestyle and academic inputs and displays a pattern-based risk estimate.
3. **Music Support** - Recommends starting genres using survey patterns from participants who reported improvement from music.
4. **Data Insights** - Presents interactive summaries from both datasets.
5. **About & Safety** - Documents the team, modeling approach, limitations, privacy, and responsible-use principles.

## Technology

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib

## Project structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   └── styles.css
├── data/
│   ├── README.md
│   ├── burnout_dashboard_sample.csv
│   └── music_survey_cleaned.csv
├── models/
│   ├── burnout_model.joblib
│   └── model_metadata.json
└── utils/
    └── recommendations.py
```

## Run the app locally

### 1. Create a virtual environment

Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## Model training

A trained demonstration model is already included. To reproduce training:

1. Place the full cleaned burnout dataset at:

```text
data/student_burnout_cleaned_full.csv
```

2. Run:

```bash
python train_model.py
```

The script saves the trained pipeline to `models/burnout_model.joblib`.

### Leakage prevention

The following columns are excluded because they directly or indirectly reveal the target label:

- `burnout_score`
- `mental_health_index`
- `dropout_risk`
- `risk_level_encoded`

The train/test split is stratified, and class weighting is used because the risk categories are imbalanced.

## Demonstration model results

| Metric | Result |
|---|---:|
| Test accuracy | 80.4% |
| Macro F1 | 75.5% |
| High-risk recall | 79.5% |

These results apply to one reproducible training run and should not be treated as clinical validation.

## Privacy and responsible use

- The current app does not save check-in responses.
- The model output is a pattern estimate, not a diagnosis.
- Self-reported data may contain sampling, cultural, and measurement bias.
- Music affects people differently, so recommendations are exploratory.
- Students should seek trusted human support when they feel overwhelmed or unsafe.

## Deployment

The app can be deployed through Streamlit Community Cloud after the repository is pushed to GitHub. Set the main file path to:

```text
app.py
```

## Future improvements

- Validate the model on additional student populations
- Add accessible language and keyboard navigation testing
- Add opt-in feedback without collecting identifying information
- Improve music recommendations with similarity modeling
- Add multilingual support
