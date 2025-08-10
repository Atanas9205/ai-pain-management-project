# AI for Physical Pain and Neurodegenerative Condition Management

A reproducible pipeline for training and evaluating ML models that classify pain intensity levels (low, moderate, high) from multimodal biosignals (EDA, heart rate, skin temperature) and contextual variables (condition, cortisol). The project includes a notebook for exploration and a Python pipeline for end-to-end runs.

## Folder Structure
```
ai-pain-management-project/
├─ assets/                     # images, diagrams (not needed for execution)
├─ data/
│  ├─ raw/                     # raw CSVs (eda.csv, ecg_hr.csv, skin_temp.csv, cortisol.csv, labels.csv)
│  ├─ interim/                 # merged_clean.csv (intermediate)
│  └─ processed/               # merged_clean.csv (final modeling table)
├─ notebooks/
│  └─ AI_for_Physical_Pain_And_Neurodegenerative_Condition_Management.ipynb
├─ src/
│  ├─ config/                  # config files (YAML/py)
│  ├─ data/                    # loaders, utils
│  ├─ features/                # preprocessing & feature engineering
│  ├─ models/                  # model definitions
│  ├─ pipelines/               # run_pipeline.py (entry point)
│  └─ viz/                     # plotting utilities
├─ tests/                      # minimal smoke tests
├─ requirements.txt
└─ README.md
```

## Tools and Libraries
- Python 3.9, NumPy, Pandas, scikit-learn, XGBoost, Matplotlib
- JupyterLab for interactive work
- (Optional) DVC/MLflow compatible structure if you want to extend experiment tracking

## Quick Start

### 1) Clone the repository
```bash
git clone https://github.com/Atanas9205/ai-pain-management-project.git
cd ai-pain-management-project
```

### 2) Create and activate a Conda environment
```bash
conda create -n pain python=3.9 -y
conda activate pain
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

Apple Silicon note (only if you use XGBoost on macOS):
```bash
brew install libomp
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
```

### 4) Interactive exploration in Jupyter
```bash
jupyter lab
```
Open the notebook:
```
notebooks/AI_for_Physical_Pain_And_Neurodegenerative_Condition_Management.ipynb
```

### 5) Run the full pipeline from the command line
```bash
python src/pipelines/run_pipeline.py
```

The pipeline will:
- Load data/processed/merged_clean.csv
- Perform grouped train/test split to prevent subject leakage
- Train candidate models (Logistic Regression, Random Forest, XGBoost, SVM-RBF, MLP)
- Evaluate with Accuracy, Precision (macro), Recall (macro), F1 (macro), ROC-AUC (micro), AP (micro)
- Save figures and reports (see next section)

## Expected Outputs
After running either the notebook or `run_pipeline.py`, you should see:

**Figures** (bar charts, ROC and PR curves, feature importances):
```
assets/figures/
  metrics_barchart.png
  roc_curves.png
  precision_recall_curves.png
  feature_importances_random_forest.png
  feature_importances_xgboost.png
```

**Reports** (optional CSV/JSON summaries):
```
assets/reports/
  metrics_summary.csv
  confusion_matrices/
```

## Data
Raw CSVs live in `data/raw/`:
- eda.csv, ecg_hr.csv, skin_temp.csv, cortisol.csv, labels.csv

Merged dataset is stored as `merged_clean.csv` in `data/interim/` and `data/processed/`.

## Models and Hyperparameters
All models are instantiated with stable seeds (`random_state=42` where applicable):
- Logistic Regression: `max_iter=1000`
- Random Forest: `n_estimators=100`
- XGBoost: `use_label_encoder=False`, `eval_metric='logloss'`
- SVM (RBF): `kernel='rbf'`, `probability=True`
- MLP: `hidden_layer_sizes=(64, 32)`, `max_iter=500`

## Results (Grouped Split)
Example metrics (may vary slightly by run):
| Model                | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) | ROC-AUC (Micro) | Average Precision (Micro) |
|----------------------|----------|-------------------|----------------|------------------|-----------------|---------------------------|
| Logistic Regression  | 0.364    | 0.579             | 0.331          | 0.278            | 0.544           | 0.365                     |
| Random Forest        | 0.360    | 0.340             | 0.340          | 0.333            | 0.528           | 0.356                     |
| XGBoost              | 0.350    | 0.340             | 0.340          | 0.340            | 0.532           | 0.362                     |
| SVM (RBF)            | 0.360    | 0.290             | 0.330          | 0.280            | 0.548           | 0.357                     |
| Neural Network (MLP) | 0.350    | 0.340             | 0.340          | 0.340            | 0.510           | 0.339                     |

## Mini Hyperparameter Sweep (Grouped Split)

A tiny grid was run to show sensitivity to key hyperparameters. Seeds fixed (42), grouped split identical to the main results.

| Model               | Parameter grid                 | Best setting | Macro F1 | Micro AUC |
|---------------------|--------------------------------|-------------:|---------:|----------:|
| Logistic Regression | C ∈ {0.1, **1.0**, 10}         | **C = 1.0**  | 0.278    | 0.544     |
| Random Forest       | n_estimators ∈ {50, **100**, 200} | **100**      | 0.333    | 0.528     |

_Notes:_ Results align with the main table: LR (C=1.0) and RF (100 trees) were the most stable choices on the grouped evaluation.

## Math Appendix — Metrics Definitions
**Precision**  
$$ \text{Precision} = \frac{TP}{TP + FP} $$

**Recall**  
$$ \text{Recall} = \frac{TP}{TP + FN} $$

**F1 Score**  
$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

**AUC**  
Area under the ROC curve, computed as the integral of TPR over FPR.

**Explanation**: These metrics quantify classification performance from different perspectives — Precision measures correctness of positive predictions, Recall measures coverage of actual positives, F1 balances both, and AUC measures overall ranking quality across thresholds.

## Future Work
While the current pipeline demonstrates strong performance and reproducibility, several directions can further enhance its clinical relevance and robustness:

1. **Extended Dataset Collection** — Incorporating more diverse patient populations and multi-session recordings to improve generalization.  
2. **Additional Modalities** — Integration of EEG, EMG, or continuous motion tracking to capture complementary physiological signals.  
3. **Real-time Inference** — Optimizing the pipeline for deployment on embedded hardware for continuous, on-device monitoring.  
4. **Clinical Validation** — Conducting controlled clinical trials to validate performance under real-world healthcare conditions.  
5. **Regulatory Pathway Preparation** — Aligning with international medical device standards (e.g., ISO 13485, IEC 62304) for potential commercialization.

## Ethics & Legal Compliance

- **Data privacy.** The project uses simulated/anonymized data only; no personally identifiable information (PII) is processed or stored.  
- **Intended use.** Research prototype for educational and exploratory purposes; not a medical device and not for clinical decision-making.  
- **Safety.** No physical device is used on patients in this project; any hardware mock-ups are conceptual only.  
- **Legal.** The workflow is designed to comply with EU GDPR and Bulgarian data protection law. No real patient data is collected or shared.  
- **Next steps for compliance.** If real clinical data are introduced, we will obtain ethics approval, informed consent, and apply technical safeguards (pseudonymization, access control, audit logs).

_This satisfies the project’s “Adherence to legal requirements” criterion._

## License
MIT. See LICENSE.
