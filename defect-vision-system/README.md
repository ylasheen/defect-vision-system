# Defect Vision System

An end-to-end **Computer Vision** system that inspects product-surface photos
and classifies them as **good** or **defective**, with **Grad-CAM
explainability** showing exactly which pixels drove the decision, and a
**business cost report** quantifying savings vs. manual inspection.

Built by **Youssef Lasheen** — AI & Machine Learning Engineer.

## Why this project is different
Most CV portfolio projects stop at "train a CNN, report accuracy." This one
goes further:

1. **CNN classifier** (TensorFlow/Keras) trained on surface-inspection images.
2. **Grad-CAM explainability** — a heatmap overlay showing *why* the model
   flagged a surface as defective (the visual equivalent of SHAP for images).
3. **MLflow experiment tracking** — every training run logs params/metrics.
4. **Business layer** — converts model accuracy into estimated **daily cost
   savings** vs. manual visual inspection on a production line.
5. **Production-ready serving** — FastAPI endpoints (including one that
   returns the Grad-CAM overlay as base64), a Streamlit live-inspection
   dashboard, Docker containerization, and automated tests.
6. **4 walkthrough notebooks** (EDA → training → Grad-CAM → evaluation),
   pre-executed with real outputs so they render directly on GitHub.

## Project structure
```
defect-vision-system/
├── data/
│   ├── raw/            # Per-class source images (good/, defective/)
│   └── processed/      # train/val/test split, ready for Keras
├── notebooks/
├── src/
│   ├── data/            # Synthetic dataset generation + train/val/test split
│   ├── models/          # CNN training, Grad-CAM, evaluation
│   ├── visualization/
│   └── utils/           # Config + logging helpers
├── models/
│   ├── saved_models/    # Trained Keras model (.keras)
│   ├── artifacts/        # Class name mapping
│   └── logs/
├── reports/
│   ├── figures/          # Confusion matrix, Grad-CAM sample sheet
│   └── summary.md        # Business cost report
├── config/config.yaml
├── tests/
├── api/main.py           # FastAPI serving layer
├── dashboard/app.py      # Streamlit live-inspection dashboard
├── Dockerfile
├── runtime.txt           # Pins Python 3.11 for cloud deployment
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# 1. Generate the synthetic inspection-camera image dataset
python src/data/make_dataset.py

# 2. Split into train/val/test
python src/data/prepare_splits.py

# 3. Train the CNN (~30 seconds on CPU, tracked with MLflow)
python src/models/train_model.py

# View MLflow experiment tracking UI (optional)
mlflow ui --backend-store-uri sqlite:///mlflow_runs/mlflow.db

# 4. Evaluate + generate Grad-CAM samples + business report
python src/models/evaluate_model.py

# 5. Serve predictions via API
uvicorn api.main:app --reload

# 6. (Optional) Launch the live-inspection dashboard
streamlit run dashboard/app.py
```

The trained model (`models/saved_models/defect_cnn.keras`) and class mapping
are committed to the repo, so steps 5 and 6 work immediately without
retraining — only re-run steps 1–4 if you want to regenerate the dataset or
retrain from scratch.

## Results (on the included synthetic dataset)
- **Test accuracy:** typically 97–99% (varies slightly per run due to random
  data generation and weight initialization)
- **False alarms:** consistently 0 in testing — the synthetic "good" class is
  visually uniform, so the model rarely misclassifies a clean surface
- **Estimated daily savings vs. manual inspection:** ~$2,000–2,700/day at
  500 units/day, depending on the run (see `reports/summary.md` for the
  exact figures from your own training run)

## Dataset note
Since this environment has no internet access to download a real industrial
defect dataset (e.g. MVTec AD, NEU surface defect), a **statistically
realistic synthetic dataset** is generated: a brushed-metal-style base
texture with directional noise and lighting gradients, onto which random
scratches, dents, and pit clusters are drawn for the "defective" class. The
loader (`tf.keras.utils.image_dataset_from_directory`) expects a standard
**ImageFolder-style structure** (`<split>/<class>/*.png`), so swapping in real
inspection photos is a drop-in replacement — just place them in
`data/raw/good/` and `data/raw/defective/` and re-run the pipeline.

**This download includes the generated `data/raw/` and `data/processed/`
images** so the project works immediately without regenerating anything.
If you push this to GitHub, consider keeping `data/raw/` and `data/processed/`
in `.gitignore` (already configured) — they're ~4,800 small files which can
make `git push` slower/flakier on an unstable connection. Regenerating them
locally takes about 5 seconds (`python src/data/make_dataset.py && python
src/data/prepare_splits.py`), so excluding them from git costs nothing.

## Tech stack
Python, TensorFlow/Keras, Grad-CAM, MLflow, scikit-learn, FastAPI, Streamlit,
Plotly, Docker, Pytest, Jupyter.

## Author
**Youssef Lasheen** — AI & Machine Learning Engineer
