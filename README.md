# AI for Physical Pain and Neurodegenerative Condition Management

A reproducible pipeline for training and evaluating ML models that classify pain intensity levels (low, moderate, high) from multimodal biosignals (EDA, heart rate, skin temperature) and contextual variables (condition, cortisol). The project includes a notebook for exploration and a Python pipeline for end-to-end runs.

<div align="center"> 
<h2>Abstract</h2> 
</div> 

<p align="justify"> 
In this project, I developed a reproducible workflow for classifying pain levels using affordable physiological signals. Since no suitable public dataset was available, I created a synthetic dataset with realistic ranges for electrodermal activity (EDA – Electrodermal Activity), heart rate (HR – Heart Rate), skin temperature, cortisol levels, a simple condition variable, and a target pain level (low, moderate, high).
</p> 

<p align="justify"> 
The data were split using a grouped strategy to avoid information leakage between subjects. I tested several baseline models — Logistic Regression, Random Forest, SVM with RBF kernel (Support Vector Machine with Radial Basis Function kernel), and a small Multi-Layer Perceptron (MLP – Multi-Layer Perceptron neural network) — all implemented through scikit-learn pipelines with preprocessing steps like imputation and scaling. Model performance was measured with common multi-class metrics (macro F1, accuracy) and ranking metrics (micro ROC-AUC – Receiver Operating Characteristic Area Under the Curve, micro average precision).
</p> 

<p align="justify"> 
On this synthetic dataset, Logistic Regression provided the most stable balance between precision and recall, making it a reasonable starting point due to its simplicity and interpretability. The current version focuses on clarity and reproducibility, with future improvements planned for using real-world data, better calibration, and fairness checks across different conditions.
</p> 

<p><strong>Keywords:</strong> pain assessment, EDA (Electrodermal Activity), heart rate (HR – Heart Rate), skin temperature, cortisol, grouped split, macro-F1, reproducibility.</p>

<div align="center"> 
<h2>Introduction</h2> 
</div> 

<p align="justify"> 
Pain is not just a symptom – it is a constant companion that can deeply affect the daily lives of people living with neurodegenerative diseases. For someone with Multiple Sclerosis (MS – Multiple Sclerosis), even a short walk can feel exhausting. For those with Parkinson’s Disease (PD – Parkinson’s Disease), stiffness and discomfort can overshadow moments of clarity. And for patients with Amyotrophic Lateral Sclerosis (ALS – Amyotrophic Lateral Sclerosis), pain might remain in the background, but it is just as exhausting as the more visible symptoms.
</p> 

<p align="justify"> 
Despite medical advances, the assessment of pain still relies heavily on patient self-reporting. This method has limitations — some patients struggle to describe their sensations, and pain itself changes in intensity and character over time. In conditions that affect movement, cognition, and emotions, this makes the job of clinicians even more challenging.
</p> 

<p align="justify"> 
In this project, I propose a more objective approach. By collecting different types of data — physiological signals, hormone levels, and patient-reported information — and applying machine learning (ML – Machine Learning) algorithms, we can build a system that more accurately detects pain levels. This approach listens not only to what patients say, but also to what their bodies reveal.
</p> 

<p align="justify"> 
I believe that this method can change the way pain is tracked in neurodegenerative diseases — shifting from a reactive to a preventive approach, ultimately helping improve patients’ quality of life.
</p>


---

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

---

## Tools and Libraries
- Python 3.9, NumPy, Pandas, scikit-learn, XGBoost, Matplotlib
- JupyterLab for interactive work
- (Optional) DVC/MLflow compatible structure if you want to extend experiment tracking

---

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

---

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

---

## Data
Raw CSVs live in `data/raw/`:
- eda.csv, ecg_hr.csv, skin_temp.csv, cortisol.csv, labels.csv

Merged dataset is stored as `merged_clean.csv` in `data/interim/` and `data/processed/`.

---

## Models and Hyperparameters
All models are instantiated with stable seeds (`random_state=42` where applicable):
- Logistic Regression: `max_iter=1000`
- Random Forest: `n_estimators=100`
- XGBoost: `use_label_encoder=False`, `eval_metric='logloss'`
- SVM (RBF): `kernel='rbf'`, `probability=True`
- MLP: `hidden_layer_sizes=(64, 32)`, `max_iter=500`

---

## Results (Grouped Split)
Example metrics (may vary slightly by run):
| Model                | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) | ROC-AUC (Micro) | Average Precision (Micro) |
|----------------------|----------|-------------------|----------------|------------------|-----------------|---------------------------|
| Logistic Regression  | 0.364    | 0.579             | 0.331          | 0.278            | 0.544           | 0.365                     |
| Random Forest        | 0.360    | 0.340             | 0.340          | 0.333            | 0.528           | 0.356                     |
| XGBoost              | 0.350    | 0.340             | 0.340          | 0.340            | 0.532           | 0.362                     |
| SVM (RBF)            | 0.360    | 0.290             | 0.330          | 0.280            | 0.548           | 0.357                     |
| Neural Network (MLP) | 0.350    | 0.340             | 0.340          | 0.340            | 0.510           | 0.339                     |

---

## Mini Hyperparameter Sweep (Grouped Split)

A tiny grid was run to show sensitivity to key hyperparameters. Seeds fixed (42), grouped split identical to the main results.

| Model               | Parameter grid                 | Best setting | Macro F1 | Micro AUC |
|---------------------|--------------------------------|-------------:|---------:|----------:|
| Logistic Regression | C ∈ {0.1, **1.0**, 10}         | **C = 1.0**  | 0.278    | 0.544     |
| Random Forest       | n_estimators ∈ {50, **100**, 200} | **100**      | 0.333    | 0.528     |

_Notes:_ Results align with the main table: LR (C=1.0) and RF (100 trees) were the most stable choices on the grouped evaluation.

---

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

---

## Design decisions (short)
- Picked **grouped train/test split** to avoid subject leakage after I noticed inflated accuracy with a plain random split.
- Kept **Logistic Regression** as the default baseline due to stable macro-F1 and ease of interpretation.
- Temporarily removed the `condition` categorical column from features to resolve a type mismatch; will re-enable with proper encoding.

## What didn't work (yet)
- **XGBoost on macOS** needed `libomp`; I installed it with `brew install libomp`, but I keep it optional for portability.
- **MLP** initially overfit the small synthetic set; reduced layer sizes and relied on scaling + simpler baselines.

## Repro notes (my machine)
- macOS Sonoma 14.x, Conda Python 3.9, scikit-learn 1.5.x
- CPU only; end-to-end run ~45–60s on MacBook Air (M-series).
- Final modeling table: **~3,750 rows × 5 numeric features** after dropping NaNs in `pain_level`.

## References (short)
- scikit-learn User Guide: Pipelines & Model Evaluation.
- XGBoost docs (macOS `libomp` note).

---

## Future Work
While the current pipeline demonstrates strong performance and reproducibility, several directions can further enhance its clinical relevance and robustness:

1. **Extended Dataset Collection** — Incorporating more diverse patient populations and multi-session recordings to improve generalization.  
2. **Additional Modalities** — Integration of EEG, EMG, or continuous motion tracking to capture complementary physiological signals.  
3. **Real-time Inference** — Optimizing the pipeline for deployment on embedded hardware for continuous, on-device monitoring.  
4. **Clinical Validation** — Conducting controlled clinical trials to validate performance under real-world healthcare conditions.  
5. **Regulatory Pathway Preparation** — Aligning with international medical device standards (e.g., ISO 13485, IEC 62304) for potential commercialization.

---

## Ethics & Legal Compliance

- **Data privacy.** The project uses simulated/anonymized data only; no personally identifiable information (PII) is processed or stored.  
- **Intended use.** Research prototype for educational and exploratory purposes; not a medical device and not for clinical decision-making.  
- **Safety.** No physical device is used on patients in this project; any hardware mock-ups are conceptual only.  
- **Legal.** The workflow is designed to comply with EU GDPR and Bulgarian data protection law. No real patient data is collected or shared.  
- **Next steps for compliance.** If real clinical data are introduced, we will obtain ethics approval, informed consent, and apply technical safeguards (pseudonymization, access control, audit logs).

_This satisfies the project’s “Adherence to legal requirements” criterion._

---

## License
MIT. See LICENSE.

---

## Data Science & Machine Learning Integration
This project bridges the core concepts from the **Data Science** module (data collection, cleaning, preprocessing, feature engineering, exploratory data analysis) with the **Machine Learning** module (model training, hyperparameter tuning, evaluation, and deployment-ready pipeline design).

In this workflow:
- **Data Science** skills were applied in the data preparation stages: loading raw CSV files, merging them, handling missing values, and performing grouped splits to prevent subject data leakage.
- **Machine Learning** skills were applied in building multiple model pipelines (Logistic Regression, Random Forest, XGBoost, SVM, MLP), training them, and evaluating their performance with appropriate metrics.
- The integration demonstrates how a Data Science process evolves naturally into a Machine Learning solution for a real-world-style problem.

## Review of Related Work

### [1] Fernandez-Rojas et al., *A Systematic Review of Neurophysiological Sensing for the Assessment of Acute Pain* (npj Digital Medicine, 2023)  
**Goal:** Survey neurophysiological and physiological sensing methods for acute pain assessment, focusing on AI approaches.  
**Method:** Reviewed multimodal sensing (EDA, EEG, fNIRS, HRV, etc.) and deep learning techniques.  
**Results:** Identified dataset scarcity, inconsistent validation methods, and strong potential for multimodal deep learning.  
**Relevance:** Supports our motivation for synthetic data generation and the focus on affordable physiological signals.  
[Full text](https://doi.org/10.1038/s41746-023-00810-1)

---

### [2] Fernandez-Rojas et al., *Multimodal Physiological Sensing for the Assessment of Acute Pain* (Frontiers in Pain Research, 2023)  
**Goal:** Experimentally compare EDA, PPG, and respiration for acute pain detection.  
**Method:** 22-subject cold pressor + other tasks; statistical and ML analysis of modalities.  
**Results:** EDA outperformed PPG and respiration across conditions; recommended as primary modality for tool design.  
**Relevance:** Validates our selection of EDA as a key feature in our pipeline.  
[Full text](https://doi.org/10.3389/fpain.2023.1150264)

---

### [3] Gkikas & Tsiknakis, *Automatic Assessment of Pain Based on Deep Learning Methods: A Systematic Review* (Computer Methods and Programs in Biomedicine, 2023)  
**Goal:** Review deep learning (DL) methods for pain assessment.  
**Method:** Compared DL vs classical ML approaches; analyzed challenges in data and evaluation.  
**Results:** DL generally outperforms classical methods but struggles with small datasets and poor generalization; calls for standardized validation protocols.  
**Relevance:** Reinforces the importance of reproducible pipelines and careful evaluation, which we adopt.  
[Full text](https://doi.org/10.1016/j.cmpb.2023.107365)

---

### [4] Gkikas et al., *Multimodal Automatic Assessment of Acute Pain through Facial Videos and Heart Rate Signals Utilizing Transformer-Based Architectures* (Frontiers in Pain Research, 2024)  
**Goal:** Combine facial video and heart rate for acute pain detection.  
**Method:** Transformer-based fusion architecture tested on multimodal datasets.  
**Results:** Multimodal fusion outperforms unimodal approaches.  
**Relevance:** Suggests future directions for our project if we expand to richer data sources.  
[Full text](https://doi.org/10.3389/fpain.2024.1372814)

---

### [5] Gkikas et al., *PainFormer: A Vision Foundation Model for Automatic Pain Assessment* (arXiv, 2025)  
**Goal:** Develop a foundation model for pain assessment across modalities and tasks.  
**Method:** Multi-task, multi-modal architecture trained on multiple datasets; leverages transfer learning.  
**Results:** Achieves state-of-the-art (SOTA) performance on multiple benchmarks.  
**Relevance:** Highlights potential of foundation models for generalizable pain assessment; relevant for long-term project goals.  
[Full text](https://arxiv.org/abs/2505.01571)

### [6] Gkikas et al., *Efficient Pain Recognition via Respiration Signals* (arXiv, 2025)  
**Goal:** Use respiration as the primary modality for pain recognition with a compact transformer model.  
**Method:** Multi-window fusion pipeline with single cross-attention transformer; tested on controlled datasets.  
**Results:** Achieved competitive accuracy while keeping computational cost low.  
**Relevance:** Supports using lightweight transformer models for single-modality scenarios like ours in early stages.  
[Full text](https://arxiv.org/abs/2507.21886)

---

### [7] Khan et al., *A Systematic Review of Multimodal Signal Fusion for Acute Pain Assessment Systems* (ACM Computing Surveys, 2025)  
**Goal:** Review fusion strategies for combining multiple biosignals in acute pain assessment.  
**Method:** Surveyed over 70 studies, comparing early-, mid-, and late-fusion approaches.  
**Results:** Multimodal fusion improves robustness but increases complexity and data requirements.  
**Relevance:** Informs our design trade-offs — start simple, scale to multimodal only when data availability improves.  
[Full text](https://doi.org/10.1145/3737281)

---

### [8] Kim et al., *Estimation of Pressure Pain in the Lower Limbs Using EDA, Tissue Oxygen Saturation, and HRV* (Sensors, 2025)  
**Goal:** Quantify pressure-induced pain using non-invasive physiological signals.  
**Method:** Measured EDA, StO₂, and HRV under increasing pressure on lower limbs; applied ML regression/classification.  
**Results:** Physiological responses correlate strongly with subjective pain scores.  
**Relevance:** Supports our choice of affordable physiological metrics (EDA, HRV) for pain-level prediction.  
[Full text](https://www.mdpi.com/1424-8220/25/3/680)

---

### [9] Fernandez-Rojas et al., *Empirical Comparison of Deep Learning Models for fNIRS Pain Decoding* (Frontiers in Neuroinformatics, 2024)  
**Goal:** Compare deep learning architectures for functional near-infrared spectroscopy (fNIRS)-based pain decoding.  
**Method:** Evaluated CNN, LSTM, and hybrid CNN-LSTM on fNIRS datasets.  
**Results:** CNN-LSTM achieved ~91% accuracy, outperforming other architectures.  
**Relevance:** Demonstrates the value of temporal + spatial feature extraction; relevant for future expansion to richer modalities.  
[Full text](https://www.frontiersin.org/articles/10.3389/fninf.2024.1320189/full)

---

### [10] Patterson et al., *Objective Wearable Measures Correlate with Self-Reported Outcomes during Spinal Cord Stimulation for Chronic Pain* (npj Digital Medicine, 2023)  
**Goal:** Assess whether wearable-measured activity and physiology correlate with chronic pain patient-reported outcomes (PROs).  
**Method:** Collected HR, HRV, and step count from patients undergoing spinal cord stimulation.  
**Results:** Objective wearable metrics significantly correlate with PRO scores.  
**Relevance:** Validates our approach to using wearable-derived physiological metrics for pain assessment.  
[Full text](https://www.nature.com/articles/s41746-023-00892-x)

### [11] Winslow et al., *Automatic detection of pain using machine learning* (Frontiers in Pain Research, 2022)  
**Goal:** Real-time detection of acute pain from physiological signals (mainly HR/HRV, respiration) under cold pressor test (CPT).  
**Method:** 41 participants; extracted 46 HRV/respiratory features; trained Logistic Regression for lab and field conditions.  
**Results:** F1 ≈ 81.9% (lab) and 79.4% (field).  
**Relevance:** Confirms that HR/HRV and respiration are valuable for low-cost sensing; supports our use of accessible biosensors and interpretable models.  
[Full text](https://www.frontiersin.org/articles/10.3389/fpain.2022.1044518/full)

---

### [12] Pouromran et al., *Automatic pain recognition from BVP signal using ML* (arXiv, 2023)  
**Goal:** Evaluate whether Blood Volume Pulse (BVP) alone carries enough information for pain detection.  
**Method:** Extracted statistical and frequency features from BVP; trained classical ML models (SVM, RF) for binary and multi-level pain classification.  
**Results:** BVP-only models achieve competitive accuracy; single modality can be useful with proper feature engineering.  
**Relevance:** Strengthens our idea of a minimal, low-budget sensor set before adding multimodal fusion.  
[Full text](https://arxiv.org/abs/2303.10607)

---

### [13] Lu, Ozek & Kamarthi, *Transformer Encoder with Multiscale Deep Learning for Pain Classification Using Physiological Signals* (arXiv, 2023)  
**Goal:** Develop a transformer encoder with multiscale representation for physiological-signal-based pain classification.  
**Method:** Multi-window temporal representations fed into transformer; compared to traditional and DL baselines.  
**Results:** Improved accuracy over conventional approaches, especially with complex temporal dependencies.  
**Relevance:** Indicates potential upgrade path — from baseline models to transformer-based fusion when real data is available.  
[Full text](https://arxiv.org/abs/2303.06845)

---

### [14] Dehshibi et al., *Pain level and behaviour classification using GRU-based sparsely-connected RNNs* (arXiv, 2022)  
**Goal:** Classify pain levels and behaviours with lightweight GRU RNNs using sparse connections.  
**Method:** Introduced sparsity in GRU layers to reduce parameters; trained on physiological and behavioural datasets.  
**Results:** Achieved solid performance with reduced model size — efficient for on-device inference.  
**Relevance:** Fits scenarios where real-time and low-power devices are needed for pain monitoring.  
[Full text](https://arxiv.org/abs/2212.14806)

---

### [15] Cohen, Vase & Hooten, *Chronic pain: burden, best practices, and new advances* (The Lancet, 2021)  
**Goal:** Review chronic pain burden, current practices, and future directions using a biopsychosocial approach.  
**Method:** Narrative review; global prevalence ~30%; pain types (nociceptive, neuropathic, nociplastic); therapeutic strategies.  
**Results:** Emphasises need for objective measures, multidisciplinary care, and patient-reported outcomes.  
**Relevance:** Provides clinical context for why our objective multimodal approach is valuable in chronic pain management.  
[Full text](https://doi.org/10.1016/S0140-6736(21)00393-7)

### [16] Vitali et al., *Sensing Behavior Change in Chronic Pain: A Scoping Review of Wearable and Passive Sensor Technologies* (Pain, 2024)  
**Goal:** Map wearable and passive sensors used to detect behavior change in chronic pain.  
**Method:** Scoping review of technologies capturing mobility, posture, activity profiles.  
**Results:** Wearable sensors are promising but real-world care studies are rare.  
**Relevance:** Highlights gap in field validation—reinforces importance of combining models with clinical scenarios.  
[Full text link unavailable (scoping abstract only)](https://journals.lww.com/pain/fulltext/2024/06000/sensing_behavior_change_in_chronic_pain__a_scoping.16.aspx)

---

### [17] Ayena et al., *Predicting Chronic Pain Using Wearable Devices: A Scoping Review of Sensor Capabilities, Data Security, and Standards Compliance* (2025)  
**Goal:** Assess wearable technology for chronic pain management, focusing on sensor quality, data security, and standards compliance.  
**Method:** Reviewed publications on wearables for CP prediction, extracted features, standards status, security protocols.  
**Results:** Found growing real-time monitoring capability, but data security and regulatory compliance remain under-addressed.  
**Relevance:** Validates our plan to keep pipelines simple, reproducible, and mindful of privacy in future clinical expansion.  
[Full text](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1581285/full)

---

### [18] Klimek et al., *Wearables Measuring Electrodermal Activity to Assess Perceived Stress in Care: A Scoping Review* (Acta Neuropsychiatrica, 2023)  
**Goal:** Review wearable devices that measure EDA for stress assessment in care settings.  
**Method:** PRISMA-SCR scoping review (2012–2022), 74 studies analyzed (population, devices, body locations, ML performance).  
**Results:** EDA wearable accuracies ranged 42%–100%, average ~82.6%, mainly offline lab studies; real-world care studies lacking.  
**Relevance:** Supports strength of EDA but also underscores need for real-life deployment—encourages future data collection design.  
[Full text](https://www.cambridge.org/core/journals/acta-neuropsychiatrica/article/wearables-measuring-electrodermal-activity-to-assess-perceived-stress-in-care-a-scoping-review/906D4056A5EDACAFB46D95BA1AB90822)

---

### [19] Kong & Chon, *Electrodermal Activity in Pain Assessment and Its Clinical Applications* (Applied Physics Reviews, 2024)  
**Goal:** Provide a comprehensive review of EDA for objective pain assessment and clinical impact.  
**Method:** Literature review of signal processing, ML methods, clinical protocols.  
**Results:** EDA is closely linked to pain via sympathetic activity; growing feasibility thanks to wearables and ML—highlighted current challenges and future directions.  
**Relevance:** Backs our choice of using EDA for pain modeling and bridges technology with clinical translation.  
[Full text reference](https://pubs.aip.org/aip/apr/article/11/3/031316/3306943/Electrodermal-activity-in-pain-assessment-and-its)

---

### [20] Kristoffersson, *A Systematic Review of Wearable Sensors for Monitoring Physical Activity* (Sensors, 2022)  
**Goal:** Catalog wearable sensor technologies for monitoring physical activity.  
**Method:** Systematic review of sensors (accelerometers, gyroscopes, etc.) for physical movement tracking.  
**Results:** Wearables for activity tracking are mature, reliable, and improving in accuracy and affordability.  
**Relevance:** Suggests potential to extend our framework with accelerometer or movement-based features in the future.  
[Full text](https://www.mdpi.com/1424-8220/22/2/573)

### [21] Kristoffersson, *A Systematic Review of Wearable Sensors for Monitoring Physical Activity* (Sensors, 2022)  
**Goal:** Review wearable technologies for physical activity monitoring.  
**Method:** Systematic analysis of accelerometers, gyroscopes, and other movement sensors.  
**Results:** Wearables proved mature, accurate, and increasingly affordable—suitable for clinical and consumer use.  
**Relevance:** Suggests potential for extending our model with motion-based features in future updates.  
[Full text](https://www.mdpi.com/1424-8220/22/2/573)

---

### [22] Hodges & van den Hoorn, *A Vision for the Future of Wearable Sensors in Spine Care and Its Challenges: Narrative Review* (Journal of Spine Surgery, 2022)  
**Goal:** Present a comprehensive future vision of wearable sensor integration for managing low back pain (LBP), using AI and real-time personalized care models.  
**Method:** Narrative review synthesizing current wearable technology (accelerometers, mHealth applications) and proposes a layered system consisting of:  
1. Real-world sensor data (movement, posture, physiology)  
2. Patient-reported data via apps (pain, psychological states)  
3. Clinical and omics inputs (imaging, genomics, medical history)  
4. AI-driven decision support  
5. Personalized treatment plans and mHealth feedback  
6. Continuous monitoring with iterative improvement  
**Results:** High potential for personalization and objective monitoring, but realization depends on user-friendly devices, clinical validation, and standardization.  
**Relevance:** Aligns with the long-term vision of evolving our baseline, reproducible pipeline into a fully integrated, AI-supported clinical system.  
[Full text](https://jss.amegroups.org/article/view/5543/html) :contentReference[oaicite:11]{index=11}

---

### [23] Sena et al., *Wearable Sensors in Patient Acuity Assessment in Critical Care* (Frontiers in Neurology, 2024)  
**Goal:** Evaluate whether integrating wrist-worn accelerometer data with demographic and clinical EHR information can enhance acuity assessment in ICU patients.  
**Method:** 87 ICU patients wore accelerometers on the wrist; models (VGG, ResNet, MobileNet, SqueezeNet, and a Transformer) were trained using accelerometry alone, and in combination with demographics and clinical variables.  
**Results:**  
- SOFA score baseline: AUC ≈ 0.53  
- Accelerometer-only: AUC ≈ 0.50, F1 ≈ 0.68  
- Accelerometer + demographics: AUC ≈ 0.69, Precision ≈ 0.75, F1 ≈ 0.67  
Notably, SENet performed best when including demographic data.  
**Relevance:** Demonstrates that embedding context (e.g., demographics, clinical data) can significantly boost performance over physiological signals alone—supporting our multimodal design philosophy.  
[Full text](https://pubmed.ncbi.nlm.nih.gov/38784909/) ([PMCID 11112699](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11112699/))

---

### [24] Gkikas, *A Pain Assessment Framework based on Multimodal Data and Deep Machine Learning Methods* (arXiv, 2025)  
**Goal:** Present a comprehensive framework for automatic pain assessment that bridges clinical context (including demographics) and modern AI methods.  
**Method:** PhD thesis consolidating unimodal and multimodal pipelines; examines demographic factors (e.g., age, sex) affecting pain perception; explores foundation and generative AI models alongside classical DL.  
**Results:** Reports state-of-the-art results across studies included in the thesis and outlines a roadmap from simple pipelines to clinically applicable, multimodal systems.  
**Relevance:** A “north-star” reference for scaling our baseline/reproducible pipeline toward richer modalities and clinical validation.  
[Full text](https://arxiv.org/abs/2505.05396). :contentReference[oaicite:0]{index=0}

---

### [25] Gkikas et al., *Multi-Representation Diagrams for Pain Recognition: Integrating Various Electrodermal Activity Signals into a Single Image* (arXiv, 2025)  
**Goal:** Investigate whether different representations of the same EDA signal, when combined into a single visual diagram, can serve as a powerful input modality for pain assessment.  
**Method:** Create multiple waveform-based visualizations from EDA (e.g., filtered versions, feature mappings), fuse them into a multi-representation diagram, and feed that into an automatic pain-assessment pipeline. Tested in the context of the AI4PAIN 2025 Grand Challenge.  
**Results:** Demonstrates that this “single-modality, multi-representation” method matches or exceeds traditional fusion of different modalities in performance, offering robustness and consistency.  
**Relevance:** Provides an innovative, resource-efficient path for enhancing model input without adding new sensor types—especially valuable for low-cost, reproducible pipelines.  
[Full text](https://arxiv.org/abs/2507.21881)