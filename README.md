# CodeAlpha Machine Learning Tasks

Three standalone Python exercises from the CodeAlpha internship:

| Script | Exercise |
| --- | --- |
| `credit_scoring.py` | Train and compare credit classification models; save and load a model for prediction |
| `emotion_recognition.py` | Emotion recognition experiment |
| `handwritten_recognition.py` | Handwritten recognition experiment |

## Credit scoring quick start

```bash
git clone https://github.com/awaisjanicode/CodeAlpha_Task-1-Credit-Scoring-Model.git
cd CodeAlpha_Task-1-Credit-Scoring-Model
python -m venv .venv
# Activate .venv for your shell
pip install numpy pandas scikit-learn joblib
python credit_scoring.py --train
python credit_scoring.py --predict --sample "age=30,income=50000,loan_amount=5000,loan_duration_months=24,num_credit_lines=2,delinquencies=0"
```

If `data/credit_data.csv` is absent, the credit script generates a **synthetic** dataset. The trained `credit_scoring_model.joblib` is generated locally. Review each other script's imports and input requirements before running it; this repository does not contain a shared `requirements.txt`.

These are educational experiments. Synthetic credit labels and model predictions must not be used to make real lending decisions.
