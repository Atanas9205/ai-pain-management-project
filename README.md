# AI-Based Physical Pain Management in Neurodegenerative Disorders

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible pipeline for training and evaluating ML models that classify pain intensity levels (low, moderate, high) from multimodal biosignals (EDA, heart rate, skin temperature) and contextual variables (condition, cortisol). The project includes a notebook for exploration and a Python pipeline for end-to-end runs.

---

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models & Methods](#models--methods)
- [Results](#results)
- [Literature Review](#literature-review)
- [Future Work](#future-work)
- [Ethics & Legal Compliance](#ethics--legal-compliance)
- [License](#license)

---

## Abstract

In this project, I developed a reproducible workflow for classifying pain levels using affordable physiological signals. Since no suitable public dataset was available, I created a synthetic dataset with realistic ranges for electrodermal activity (EDA – Electrodermal Activity), heart rate (HR – Heart Rate), skin temperature, cortisol levels, a simple condition variable, and a target pain level (low, moderate, high).

The data were split using a grouped strategy to avoid information leakage between subjects. I tested several baseline models – Logistic Regression, Random Forest, SVM with RBF kernel (Support Vector Machine with Radial Basis Function kernel), and a small Multi-Layer Perceptron (MLP – Multi-Layer Perceptron neural network) – all implemented through scikit-learn pipelines with preprocessing steps like imputation and scaling. Model performance was measured with common multi-class metrics (macro F1, accuracy) and ranking metrics (micro ROC-AUC – Receiver Operating Characteristic Area Under the Curve, micro average precision).

On this synthetic dataset, Logistic Regression provided the most stable balance between precision and recall, making it a reasonable starting point due to its simplicity and interpretability. The current version focuses on clarity and reproducibility, with future improvements planned for using real-world data, better calibration, and fairness checks across different conditions.

**Keywords:** pain assessment, EDA (Electrodermal Activity), heart rate (HR – Heart Rate), skin temperature, cortisol, grouped split, macro-F1, reproducibility.

---

## Introduction

Pain is not just a symptom – it is a constant companion that can deeply affect the daily lives of people living with neurodegenerative diseases. For someone with Multiple Sclerosis (MS – Multiple Sclerosis), even a short walk can feel exhausting. For those with Parkinson's Disease (PD – Parkinson's Disease), stiffness and discomfort can overshadow moments of clarity. And for patients with Amyotrophic Lateral Sclerosis (ALS – Amyotrophic Lateral Sclerosis), pain might remain in the background, but it is just as exhausting as the more visible symptoms.

Despite medical advances, the assessment of pain still relies heavily on patient self-reporting. This method has limitations – some patients struggle to describe their sensations, and pain itself changes in intensity and character over time. In conditions that affect movement, cognition, and emotions, this makes the job of clinicians even more challenging.

In this project, I propose a more objective approach. By collecting different types of data – physiological signals, hormone levels, and patient-reported information – and applying machine learning (ML – Machine Learning) algorithms, we can build a system that more accurately detects pain levels. This approach listens not only to what patients say, but also to what their bodies reveal.

I believe that this method can change the way pain is tracked in neurodegenerative diseases – shifting from a reactive to a preventive approach, ultimately helping improve patients' quality of life.

---

## Quick Start

### Prerequisites
- Python 3.9+
- Conda or pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Atanas9205/ai-pain-management-project.git
cd ai-pain-management-project
```

2. **Create and activate environment**
```bash
conda create -n pain python=3.9 -y
conda activate pain
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Apple Silicon note** (only if using XGBoost on macOS):
```bash
brew install libomp
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
```

### Running the Project

#### Option 1: Interactive Jupyter Notebook
```bash
jupyter lab
```
Open: `notebooks/AI_Based_Physical_Pain_Management_in_Neurodegenerative_Disorders.ipynb`

#### Option 2: Command Line Pipeline
```bash
python src/pipelines/run_pipeline.py
```

The pipeline will automatically:
- Load `data/processed/merged_clean.csv`
- Perform grouped train/test split to prevent subject leakage
- Train multiple models (Logistic Regression, Random Forest, XGBoost, SVM-RBF, MLP)
- Evaluate with comprehensive metrics
- Save figures and reports

---

## Project Structure

```
ai-pain-management-project/
├── assets/                     # Images, diagrams, outputs
│   ├── figures/               # Generated plots and charts
│   └── reports/               # Model performance summaries
├── data/
│   ├── raw/                   # Original CSV files
│   │   ├── eda.csv           # Electrodermal activity
│   │   ├── ecg_hr.csv        # Heart rate data
│   │   ├── skin_temp.csv     # Skin temperature
│   │   ├── cortisol.csv      # Cortisol levels
│   │   └── labels.csv        # Pain level targets
│   ├── interim/              # Intermediate processing
│   └── processed/            # Final modeling dataset
├── notebooks/
│   └── AI_Based_Physical_Pain_Management_in_Neurodegenerative_Disorders.ipynb
├── src/
│   ├── config/               # Configuration files
│   ├── data/                 # Data loading utilities
│   │   └── loaders.py       # CSV loading functions
│   ├── features/             # Feature engineering
│   │   └── preprocess.py    # Data preprocessing pipeline
│   ├── models/               # ML model definitions
│   │   └── models.py        # Training and evaluation
│   ├── pipelines/            # Orchestration layer
│   │   └── run_pipeline.py  # Main execution script
│   └── viz/                  # Visualization utilities
│       └── plots.py         # Chart generation
├── tests/                    # Automated testing
│   └── test_smoke.py        # Basic functionality tests
├── requirements.txt          # Python dependencies
└── README.md
```

### Directory Details

#### Data Organization
- **`raw/`** – Original, unprocessed CSV files as initially generated
- **`interim/`** – Partially processed datasets for intermediate analysis
- **`processed/`** – Fully cleaned and merged dataset ready for modeling

#### Source Code (`src/`)
- **`data/`** – Data loading and utilities
- **`features/`** – Preprocessing and feature engineering
- **`models/`** – ML algorithms and evaluation metrics
- **`viz/`** – Plotting and visualization functions
- **`pipelines/`** – End-to-end workflow orchestration

#### Notebooks
Interactive Jupyter notebooks for:
- Exploratory Data Analysis (EDA)
- Model experimentation and comparison
- Results visualization and reporting

#### Testing
Automated smoke tests to ensure code reliability and reproducibility.

---

## Installation

### System Requirements
- **Operating System:** macOS, Linux, Windows
- **Python:** 3.9 or higher
- **Memory:** 4GB RAM minimum (8GB recommended)
- **Storage:** 500MB free space

### Dependencies

Core libraries used in this project:

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21.0 | Numerical computations |
| `pandas` | ≥1.3.0 | Data manipulation |
| `scikit-learn` | ≥1.5.0 | Machine learning algorithms |
| `xgboost` | ≥1.6.0 | Gradient boosting |
| `matplotlib` | ≥3.5.0 | Data visualization |
| `seaborn` | ≥0.11.0 | Statistical plotting |
| `jupyter` | ≥1.0.0 | Interactive notebooks |

### Installation Methods

#### Method 1: Conda (Recommended)
```bash
conda create -n pain python=3.9
conda activate pain
pip install -r requirements.txt
```

#### Method 2: Virtual Environment
```bash
python -m venv pain-env
source pain-env/bin/activate  # On Windows: pain-env\Scripts\activate
pip install -r requirements.txt
```

### Verification
Test your installation:
```bash
python -c "import pandas, sklearn, xgboost; print('Installation successful!')"
```

---

## Usage

### Running Tests

Verify everything works correctly:
```bash
pytest tests/ -v
```

Expected output:
```
============================= test session starts ==============================
platform darwin -- Python 3.9.23, pytest-8.4.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /path/to/ai-pain-management-project
plugins: anyio-4.9.0
collected 2 items                                                              

tests/test_smoke.py::test_schema_and_min_rows PASSED                     [ 50%]
tests/test_smoke.py::test_target_values_and_numeric_types PASSED         [100%]

============================== 2 passed in 0.58s ===============================
```

### Expected Outputs

After running the pipeline, you should see:

#### Figures
```
assets/figures/
├── metrics_barchart.png              # Model comparison chart
├── roc_curves.png                    # ROC analysis
├── precision_recall_curves.png       # PR curves
├── feature_importances_random_forest.png
└── feature_importances_xgboost.png
```

#### Reports
```
assets/reports/
├── metrics_summary.csv               # Performance metrics
└── confusion_matrices/               # Model confusion matrices
```

### Pipeline Execution Time
- **Typical runtime:** 45-60 seconds on MacBook Air (M-series)
- **Dataset size:** ~3,750 rows × 5 numeric features
- **Processing:** CPU-only, single-threaded

---

## Data

### Dataset Overview

Our synthetic dataset simulates realistic physiological responses to pain in neurodegenerative conditions:

| Signal Type | Features | Physiological Basis |
|-------------|----------|-------------------|
| **EDA** | Electrodermal activity | Sympathetic nervous system response |
| **Heart Rate** | BPM measurements | Cardiovascular stress indicators |
| **Skin Temperature** | Surface temperature | Thermoregulatory changes |
| **Cortisol** | Stress hormone levels | HPA axis activation |
| **Pain Level** | Low/Moderate/High | Target classification |

### Data Characteristics

- **Total samples:** ~3,750 rows
- **Features:** 5 numeric variables
- **Target classes:** 3 (balanced distribution)
- **Missing values:** Handled via imputation
- **Split strategy:** Grouped by subject to prevent leakage

### Data Processing Pipeline

1. **Loading** – Read individual CSV files from `data/raw/`
2. **Merging** – Combine all signals by timestamp/subject
3. **Cleaning** – Handle missing values and outliers
4. **Feature Engineering** – Scale and normalize signals
5. **Splitting** – Group-based train/test division
6. **Validation** – Ensure data quality and consistency

---
## Data

Due to privacy and legal constraints (GDPR), we cannot use real-world patient data.  
This project therefore relies on a **synthetic dataset** to demonstrate the full end-to-end machine learning pipeline.  
This approach ensures reproducibility and allows the model to be developed safely.

For a detailed explanation of the synthetic data generation process and an initial data inspection, please refer to the  
[`AI_Based_Physical_Pain_Management_in_Neurodegenerative_Disorders.ipynb`](AI_Based_Physical_Pain_Management_in_Neurodegenerative_Disorders.ipynb) notebook.

---

## Models & Methods

### Algorithms Tested

| Model | Configuration | Rationale |
|-------|--------------|-----------|
| **Logistic Regression** | `max_iter=1000` | Baseline, interpretable |
| **Random Forest** | `n_estimators=100` | Ensemble robustness |
| **XGBoost** | `eval_metric='logloss'` | Gradient boosting power |
| **SVM (RBF)** | `kernel='rbf'`, `probability=True` | Non-linear relationships |
| **Neural Network (MLP)** | `hidden_layers=(64,32)`, `max_iter=500` | Deep learning approach |

### Evaluation Metrics

#### Classification Metrics
- **Accuracy** – Overall correctness
- **Precision (Macro)** – Class-balanced precision
- **Recall (Macro)** – Class-balanced sensitivity
- **F1-Score (Macro)** – Harmonic mean of precision/recall

#### Ranking Metrics
- **ROC-AUC (Micro)** – Area under ROC curve
- **Average Precision (Micro)** – Area under PR curve

### Mathematical Definitions

**Precision**
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall**
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score**
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**AUC**
Area under the ROC curve, computed as the integral of TPR over FPR.

### Hyperparameter Tuning

Selected hyperparameters through minimal grid search:

| Model | Parameter Grid | Best Setting |
|-------|---------------|--------------|
| Logistic Regression | C ∈ {0.1, **1.0**, 10} | **C = 1.0** |
| Random Forest | n_estimators ∈ {50, **100**, 200} | **100** |

---

## Results

### Model Performance (Grouped Split)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) | ROC-AUC (Micro) | AP (Micro) |
|-------|----------|-------------------|----------------|------------------|-----------------|------------|
| **Logistic Regression** | **0.364** | **0.579** | **0.331** | **0.278** | **0.544** | **0.365** |
| Random Forest | 0.360 | 0.340 | 0.340 | 0.333 | 0.528 | 0.356 |
| XGBoost | 0.350 | 0.340 | 0.340 | 0.340 | 0.532 | 0.362 |
| SVM (RBF) | 0.360 | 0.290 | 0.330 | 0.280 | 0.548 | 0.357 |
| Neural Network (MLP) | 0.350 | 0.340 | 0.340 | 0.340 | 0.510 | 0.339 |

### Key Findings

1. **Logistic Regression** emerges as the most stable baseline with highest precision
2. **Random Forest** and **XGBoost** show consistent performance across metrics
3. **SVM** achieves good ROC-AUC but lower precision
4. **MLP** performance suggests need for larger datasets or regularization

### Hyperparameter Sensitivity

Mini grid search results (grouped split, fixed seeds):

| Model | Parameter | Best Value | Macro F1 | Micro AUC |
|-------|-----------|------------|----------|-----------|
| Logistic Regression | C | 1.0 | 0.278 | 0.544 |
| Random Forest | n_estimators | 100 | 0.333 | 0.528 |

---

## Literature Review

### Foundation Papers

#### Core Pain Assessment Research

**[1] Fernandez-Rojas et al., *A Systematic Review of Neurophysiological Sensing for the Assessment of Acute Pain* (npj Digital Medicine, 2023)**
- **Goal:** Survey neurophysiological sensing methods for acute pain assessment
- **Key Finding:** Dataset scarcity and strong potential for multimodal deep learning
- **Relevance:** Validates our synthetic data approach and multimodal design
- [Full text](https://doi.org/10.1038/s41746-023-00810-1)

**[2] Fernandez-Rojas et al., *Multimodal Physiological Sensing for the Assessment of Acute Pain* (Frontiers in Pain Research, 2023)**
- **Goal:** Compare EDA, PPG, and respiration for acute pain detection
- **Key Finding:** EDA outperformed other modalities across conditions
- **Relevance:** Supports our emphasis on EDA as primary feature
- [Full text](https://doi.org/10.3389/fpain.2023.1150264)

#### Machine Learning Applications

**[3] Gkikas & Tsiknakis, *Automatic Assessment of Pain Based on Deep Learning Methods: A Systematic Review* (Computer Methods and Programs in Biomedicine, 2023)**
- **Goal:** Review deep learning approaches for pain assessment
- **Key Finding:** DL outperforms classical ML but struggles with small datasets
- **Relevance:** Reinforces importance of reproducible pipelines and careful evaluation
- [Full text](https://doi.org/10.1016/j.cmpb.2023.107365)

**[11] Winslow et al., *Automatic detection of pain using machine learning* (Frontiers in Pain Research, 2022)**
- **Goal:** Real-time pain detection from physiological signals
- **Method:** 41 participants, HRV/respiratory features, Logistic Regression
- **Results:** F1 ≈ 81.9% (lab) and 79.4% (field)
- **Relevance:** Confirms HR/HRV effectiveness for accessible biosensors
- [Full text](https://www.frontiersin.org/articles/10.3389/fpain.2022.1044518/full)

### Wearable Technology & Clinical Applications

**[18] Klimek et al., *Wearables Measuring Electrodermal Activity to Assess Perceived Stress in Care* (Acta Neuropsychiatrica, 2023)**
- **Goal:** Review EDA wearables for stress assessment in care settings
- **Results:** EDA accuracies 42%-100%, average ~82.6%
- **Relevance:** Supports EDA strength while highlighting need for real-world deployment
- [Full text](https://www.cambridge.org/core/journals/acta-neuropsychiatrica/article/wearables-measuring-electrodermal-activity-to-assess-perceived-stress-in-care-a-scoping-review/906D4056A5EDACAFB46D95BA1AB90822)

**[22] Hodges & van den Hoorn, *A Vision for the Future of Wearable Sensors in Spine Care* (Journal of Spine Surgery, 2022)**
- **Goal:** Present comprehensive vision for wearable sensor integration in spine care
- **Method:** Proposes layered system: sensors → patient data → clinical inputs → AI → personalized treatment
- **Relevance:** Aligns with long-term vision for integrated clinical systems
- [Full text](https://jss.amegroups.org/article/view/5543/html)

### Emerging Technologies

**[5] Gkikas et al., *PainFormer: A Vision Foundation Model for Automatic Pain Assessment* (arXiv, 2025)**
- **Goal:** Develop foundation model for pain assessment across modalities
- **Results:** State-of-the-art performance on multiple benchmarks
- **Relevance:** Highlights potential of foundation models for generalizable pain assessment
- [Full text](https://arxiv.org/abs/2505.01571)

**[24] Gkikas, *A Pain Assessment Framework based on Multimodal Data and Deep Machine Learning Methods* (arXiv, 2025)**
- **Goal:** Comprehensive framework bridging clinical context and modern AI
- **Relevance:** "North-star" reference for scaling toward clinical validation
- [Full text](https://arxiv.org/abs/2505.05396)

### Clinical Context & Validation

**[15] Cohen, Vase & Hooten, *Chronic pain: burden, best practices, and new advances* (The Lancet, 2021)**
- **Goal:** Review chronic pain burden and therapeutic strategies
- **Key Finding:** ~30% global prevalence, emphasizes need for objective measures
- **Relevance:** Provides clinical context for objective multimodal approaches
- [Full text](https://doi.org/10.1016/S0140-6736(21)00393-7)

**[17] Ayena et al., *Predicting Chronic Pain Using Wearable Devices* (2025)**
- **Goal:** Assess wearable technology capabilities and compliance standards
- **Key Finding:** Growing monitoring capability but security/compliance under-addressed
- **Relevance:** Validates plan for privacy-mindful pipeline design
- [Full text](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1581285/full)

### Research Gaps & Opportunities

The literature review reveals several consistent themes:

1. **Dataset Scarcity** – Limited availability of high-quality, diverse pain datasets
2. **Validation Gap** – Most studies are lab-based; real-world validation needed
3. **Multimodal Potential** – Combining modalities improves robustness
4. **Regulatory Challenges** – Security, privacy, and compliance require attention
5. **Clinical Translation** – Need for standardized protocols and clinical integration

These findings directly inform our project's design decisions and future development priorities.

---

## Future Work

### Technical Enhancements

#### 1. Dataset Expansion
- **Multi-center data collection** from diverse patient populations
- **Longitudinal studies** to capture pain progression over time
- **Real-world validation** in clinical settings
- **Cross-condition studies** (MS, PD, ALS, fibromyalgia)

#### 2. Advanced Modeling
- **Deep learning architectures** for temporal pattern recognition
- **Foundation models** for cross-domain pain assessment
- **Federated learning** for privacy-preserving multi-site training
- **Uncertainty quantification** for clinical decision support

#### 3. Multimodal Integration
- **Additional signals:** EEG, EMG, continuous motion tracking
- **Environmental context:** Sleep patterns, activity levels, medication timing
- **Patient-reported outcomes:** Integration with mobile health apps
- **Genomic factors:** Personalization based on genetic pain sensitivity

#### 4. Real-time Systems
- **Edge computing** for on-device inference
- **Streaming analytics** for continuous monitoring
- **Alert systems** for pain episode prediction
- **Intervention triggers** for proactive care

### Clinical Translation

#### 1. Regulatory Pathway
- **FDA/CE marking** preparation for medical device classification
- **Clinical trial design** for efficacy and safety validation
- **Quality management systems** (ISO 13485, IEC 62304)
- **Post-market surveillance** frameworks

#### 2. Healthcare Integration
- **EHR integration** with major hospital systems
- **Clinical workflow** optimization and training
- **Provider dashboard** development
- **Patient engagement** tools and education

#### 3. Health Economics
- **Cost-effectiveness** studies vs. current standard of care
- **Reimbursement** pathway development
- **Population health** impact assessment
- **Healthcare utilization** reduction analysis

### Research Directions

#### 1. Methodological Advances
- **Causal inference** methods for treatment effect estimation
- **Fairness and bias** mitigation across demographic groups
- **Explainable AI** for clinician trust and adoption
- **Privacy-preserving** machine learning techniques

#### 2. Collaborative Research
- **Academic partnerships** for multi-disciplinary expertise
- **Industry collaboration** for technology development
- **Patient advocacy groups** for user-centered design
- **International consortiums** for data sharing standards

---

## Ethics & Legal Compliance

### Data Privacy & Security

#### Current Project
- **Synthetic data only** – No personally identifiable information (PII) processed
- **Local processing** – All computation performed on user's machine
- **No data transmission** – No external APIs or cloud services used
- **Open source** – Full transparency in methodology and code

#### Future Clinical Implementation

**Data Protection Compliance**
- **EU GDPR** compliance for European deployment
- **HIPAA compliance** for US healthcare integration
- **Bulgarian data protection law** adherence for local implementation
- **ISO 27001** information security management

**Technical Safeguards**
- **End-to-end encryption** for data transmission
- **Pseudonymization** techniques for patient privacy
- **Access control** with role-based permissions
- **Audit logging** for compliance monitoring

### Clinical Safety & Standards

#### Medical Device Considerations
- **Risk classification** under FDA/MDR frameworks
- **Clinical evidence** requirements for safety and efficacy
- **Quality management** system implementation
- **Post-market surveillance** and adverse event reporting

#### Ethical Guidelines
- **Informed consent** protocols for data collection
- **Ethics committee** approval for clinical studies
- **Patient autonomy** in data sharing decisions
- **Beneficence principle** ensuring patient benefit over harm

### Legal Framework

#### Intended Use Statement
> **This software is a research prototype for educational and exploratory purposes only. It is NOT intended for clinical decision-making, medical diagnosis, or treatment recommendations. Healthcare providers should not rely on this system for patient care decisions.**

#### Liability Considerations
- **Academic research** exemptions and limitations
- **Open source licensing** (MIT) disclaimers
- **No warranty** provisions for research software
- **User responsibility** for appropriate application

#### Regulatory Roadmap
1. **Research phase** (current) – Academic validation
2. **Pre-submission** consultation with regulatory bodies
3. **Clinical investigation** with proper oversight
4. **Marketing authorization** application and review
5. **Post-market** monitoring and updates

### Stakeholder Engagement

#### Patient Involvement
- **Patient advisory boards** for design input
- **User experience research** for accessibility
- **Community feedback** mechanisms
- **Educational resources** for informed participation

#### Healthcare Provider Engagement
- **Clinical workflow** integration studies
- **Training program** development
- **Feedback collection** for iterative improvement
- **Professional society** collaboration

---

## Data Science & Machine Learning Integration

This project demonstrates the natural progression from **Data Science** fundamentals to **Machine Learning** applications in healthcare:

### Data Science Components
- **Data Collection:** Simulated physiological sensor data generation
- **Data Cleaning:** Missing value imputation and outlier handling
- **Exploratory Analysis:** Statistical summaries and visualization
- **Feature Engineering:** Signal preprocessing and scaling
- **Data Validation:** Grouped splits to prevent subject leakage

### Machine Learning Components
- **Model Selection:** Comparison of 5 different algorithms
- **Pipeline Development:** End-to-end automated workflows
- **Evaluation:** Comprehensive metrics for multiclass classification
- **Hyperparameter Tuning:** Grid search optimization
- **Reproducibility:** Fixed random seeds and version control

### Integration Benefits
1. **Seamless Workflow** – Data preparation feeds directly into modeling
2. **Reproducible Results** – Consistent pipeline execution
3. **Scalable Design** – Easy adaptation to new datasets
4. **Clinical Relevance** – Domain-specific problem solving

---

## Design Decisions & Lessons Learned

### What Worked Well

#### Grouped Train/Test Split
- **Problem:** Initial random splits showed inflated accuracy
- **Solution:** Subject-based grouping prevents data leakage
- **Result:** More realistic performance estimates

#### Logistic Regression Baseline
- **Rationale:** Simple, interpretable, stable performance
- **Outcome:** Achieved best macro-F1 and precision balance
- **Benefit:** Easy to explain to clinical stakeholders

#### Comprehensive Evaluation
- **Metrics:** Combined classification and ranking measures
- **Visualization:** ROC/PR curves for threshold analysis
- **Reporting:** Automated summary generation

### Challenges & Solutions

#### XGBoost on macOS
- **Problem:** Missing `libomp` dependency
- **Solution:** `brew install libomp` with environment variable
- **Status:** Optional installation for portability

#### MLP Overfitting
- **Problem:** Neural network overfitted small synthetic dataset
- **Solution:** Reduced layer sizes, relied on simpler baselines
- **Learning:** Deep learning needs larger, more diverse datasets

#### Categorical Features
- **Problem:** Type mismatch with `condition` variable
- **Temporary Fix:** Removed from feature set
- **Future:** Proper encoding pipeline implementation

### Reproducibility Notes

#### Environment Specifications
- **OS:** macOS Sonoma 14.x (primary development)
- **Python:** 3.9 via Conda
- **Hardware:** MacBook Air (M-series), CPU-only
- **Runtime:** 45-60 seconds end-to-end

#### Version Control
- **Dependencies:** Pinned in `requirements.txt`
- **Seeds:** Fixed at 42 for all random operations
- **Data:** Deterministic synthetic generation
- **Results:** Consistent across multiple runs

---

## Testing & Validation

### Automated Testing

Run the test suite to verify installation and functionality:

```bash
pytest tests/ -v
```

#### Test Coverage
- **Smoke tests** – Basic functionality verification
- **Data loading** – CSV reading and processing
- **Model training** – Pipeline execution without errors
- **Output generation** – File creation and format validation

#### Continuous Integration
While not currently implemented, the project structure supports:
- **GitHub Actions** for automated testing
- **Docker containers** for environment consistency
- **Code quality** checks (linting, formatting)

### Manual Validation

#### Data Quality Checks
1. **Range validation** – Physiological signals within realistic bounds
2. **Distribution analysis** – Balanced target classes
3. **Correlation analysis** – Expected signal relationships
4. **Missing data** – Proper handling of null values

#### Model Validation
1. **Cross-validation** – K-fold validation on training set
2. **Learning curves** – Training vs. validation performance
3. **Feature importance** – Physiologically meaningful rankings
4. **Prediction analysis** – Qualitative review of model decisions

---

## References

### Core Citations

1. **Fernandez-Rojas, R., et al.** (2023). A systematic review of neurophysiological sensing for the assessment of acute pain. *npj Digital Medicine*, 6(1), 1-15. https://doi.org/10.1038/s41746-023-00810-1

2. **Gkikas, N., & Tsiknakis, M.** (2023). Automatic assessment of pain based on deep learning methods: A systematic review. *Computer Methods and Programs in Biomedicine*, 231, 107365. https://doi.org/10.1016/j.cmpb.2023.107365

3. **Cohen, S. P., Vase, L., & Hooten, W. M.** (2021). Chronic pain: an update on burden, best practices, and new advances. *The Lancet*, 397(10289), 2082-2097. https://doi.org/10.1016/S0140-6736(21)00393-7

### Technical Documentation

- **scikit-learn documentation:** https://scikit-learn.org/stable/
- **XGBoost documentation:** https://xgboost.readthedocs.io/
- **Pandas documentation:** https://pandas.pydata.org/docs/

### Dataset & Methodology References

Complete bibliography with 25 references available in the full literature review section above, covering:
- Neurophysiological pain assessment methods
- Machine learning applications in healthcare
- Wearable sensor technologies
- Clinical validation frameworks
- Regulatory and ethical considerations

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 AI Pain Management Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE