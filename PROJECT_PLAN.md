# Code Quality Prediction ML Project

## Project Overview

**Goal:** Train ML model to predict if a code repository is "high quality" or "low quality" WITHOUT executing the code.

**Input:** Source code files (Python)
**Output:** Quality score/classification (Good/Bad)

**Why it's unique:**
- No standard dataset exists—you build your own
- Combines software engineering knowledge with ML
- Real-world problem (code review automation)
- Entirely feasible on MacBook M2 + Google Colab

---

## Table of Contents

1. [Feature Engineering Strategy](#1-feature-engineering-strategy)
2. [Dataset Collection](#2-dataset-collection-strategy)
3. [Implementation Architecture](#3-implementation-architecture)
4. [Project Structure](#4-project-structure)
5. [Phase-wise Implementation](#5-phase-wise-implementation)
6. [Libraries & Dependencies](#6-libraries--dependencies)
7. [Timeline](#7-timeline--execution)
8. [Challenges & Solutions](#8-expected-challenges--solutions)

---

## 1. Feature Engineering Strategy

Extract these categories of features from code:

### Structural Features (AST-based)

- Average function length (lines)
- Average class size
- Maximum nesting depth
- Number of functions per file
- Average parameter count per function
- Code-to-comment ratio
- Cyclomatic complexity per function

### Naming Convention Features

- Variable name length (average)
- Use of descriptive names (vowel ratio in names)
- Snake_case vs camelCase consistency
- Single-letter variable count (bad practice indicator)
- Abbreviation ratio in names

### Code Pattern Features

- Try-except block count (error handling)
- Type hints usage percentage
- Docstring presence (functions with docs / total functions)
- Import statement cleanliness (organized vs scattered)
- Dead code indicators (unused variables, imports)
- Magic number count (hardcoded values)

### Modularity Features

- Circular dependency detection
- Module coupling (how many modules import from each other)
- Code duplication ratio (copy-paste detection)
- Test coverage estimation

### Maintainability Metrics

- Lines of code per file
- Number of files in repository
- Branch complexity
- Return statement count per function

---

## 2. Dataset Collection Strategy

### Label Definition (Binary Classification)

**High Quality (Label: 1)**
- GitHub repos with 500+ stars
- Active maintenance (commits in last 3 months)
- Has test directory
- Has documentation/README
- Examples: popular Python libraries, well-maintained projects

**Low Quality (Label: 0)**
- GitHub repos with <50 stars
- Abandoned (no commits in 1+ year)
- No tests
- Minimal documentation
- Examples: homework submissions, abandoned projects

### Dataset Size
- Target: 100-150 repositories (manageable on MacBook)
- Split: 80% train, 20% test
- Each repo: extract top 10-20 Python files to analyze

### How to Collect
- Option 1: Use GitHub API to scrape repos
- Option 2: Manually curate popular vs abandoned repos
- Option 3: Use Kaggle GitHub dataset (already labeled)

---

## 3. Implementation Architecture

### Core Libraries

```python
import ast                    # Parse Python code
import re                     # Pattern matching
import os                     # File operations
import pandas as pd           # Data handling
from pathlib import Path      # File paths
```

### Extraction Process

- Walk through all Python files in repository
- Parse each file using Python's AST module
- Extract structural features from AST
- Extract naming convention metrics
- Extract code pattern indicators
- Aggregate features across all files

---

## 4. Project Structure

```
code-quality-predictor/
├── data/
│   ├── raw_repos/              # Downloaded GitHub repos
│   ├── extracted_features.csv  # Feature matrix
│   └── labels.csv              # Quality labels
├── scripts/
│   ├── 01_scrape_github.py     # Collect repositories
│   ├── 02_feature_extractor.py # Extract features
│   ├── 03_data_preparation.py  # Clean & prepare data
│   └── 04_train_model.py       # ML pipeline
├── models/
│   └── quality_classifier.pkl  # Trained model
├── evaluation/
│   ├── metrics.txt             # Performance results
│   └── confusion_matrix.png    # Visualization
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA on Colab
└── PROJECT_PLAN.md             # This file
```

---

## 5. Phase-wise Implementation

### Phase 1: Data Collection (Week 1)

**Goal:** Collect 50-100 repositories (mix of high and low quality)

**Tasks:**
- Scrape high-quality repos from GitHub (500+ stars)
- Scrape low-quality repos (abandoned, <50 stars)
- Label each repository
- Store in data/raw_repos/ directory

**Deliverable:**
- data/labels.csv with repo names and quality labels

---

### Phase 2: Feature Extraction (Week 2)

**Goal:** Extract all features from collected repositories

**Tasks:**
- Implement CodeAnalyzer class
- Parse each Python file using AST
- Extract structural, naming, pattern, and maintainability features
- Handle edge cases (syntax errors, empty files)
- Aggregate features per repository

**Deliverable:**
- data/extracted_features.csv with feature matrix
- Each row = one repository
- Each column = one feature

---

### Phase 3: Data Preparation (Week 2)

**Goal:** Clean and prepare data for ML models

**Tasks:**
- Handle missing values (fill with 0 or mean)
- Normalize/standardize features
- Analyze feature distributions
- Create train-test split (80-20)
- Check for class imbalance

**Deliverable:**
- Cleaned, normalized feature matrix
- Train and test sets ready for modeling

---

### Phase 4: Model Training & Evaluation (Week 3)

**Goal:** Train and compare ML models

**Tasks:**
- Implement multiple models:
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
- Train on training set
- Evaluate on test set
- Perform cross-validation
- Select best model
- Generate evaluation metrics and visualizations

**Deliverable:**
- Trained model saved as models/quality_classifier.pkl
- evaluation/metrics.txt with accuracy, precision, recall, F1
- Confusion matrix visualization
- ROC-AUC curve

---

### Phase 5: Report Writing (Week 4)

**Goal:** Document entire project in LaTeX format

**Content:**
- Project motivation and objectives
- Literature review (code quality metrics)
- Methodology (feature extraction, model selection)
- Dataset description
- Experimental results and analysis
- Visualizations and graphs
- Conclusion and future work
- Code snippets and appendices

**Deliverable:**
- report.pdf (minimum 20 pages, Times New Roman, 12pt)

---

## 6. Libraries & Dependencies

### Core ML
```
scikit-learn        # Models, evaluation metrics
pandas              # Data manipulation
numpy               # Numerical operations
matplotlib          # Plotting
seaborn             # Advanced visualizations
```

### Code Analysis
```
ast                 # Built-in Python AST parser
radon               # Code metrics (optional)
```

### GitHub & Data Collection
```
requests            # HTTP requests
PyGithub            # GitHub API wrapper
```

### Installation
```bash
pip install scikit-learn pandas numpy matplotlib seaborn requests PyGithub
```

---

## 7. Timeline & Execution

| Phase | Duration | Tasks | Deliverable |
|-------|----------|-------|-------------|
| 1 | Week 1 | Data collection & labeling | labels.csv |
| 2 | Week 2 | Feature extraction | extracted_features.csv |
| 3 | Week 2 | Data cleaning & preparation | Train/test sets |
| 4 | Week 3 | Model training & evaluation | Trained model + metrics |
| 5 | Week 4 | Report writing | report.pdf (20+ pages) |

**Start Date:** Tomorrow (2026-02-05)
**Expected Completion:** 4 weeks

---

## 8. Expected Challenges & Solutions

| Challenge | Cause | Solution |
|-----------|-------|----------|
| Syntax errors in code | Some repos have broken Python | Use try-catch blocks, skip malformed files |
| Parsing timeouts | Large repos with many files | Set file/repo size limits |
| Imbalanced dataset | More high-quality than low-quality repos | Use class weights or SMOTE |
| Feature scaling issues | Different ranges | Use StandardScaler normalization |
| Model overfitting | Too many features, small dataset | Use cross-validation, regularization |
| Low model accuracy | Features not discriminative | Add sophisticated features, ensemble methods |
| Memory issues on MacBook | Processing 100+ repos simultaneously | Process repos in batches, use Colab |

---

## 9. Key Metrics & Evaluation

### Classification Metrics
- Accuracy: Overall correctness
- Precision: Of predicted high-quality, how many are correct
- Recall: Of actual high-quality, how many did we find
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Performance across all thresholds

### Feature Importance
- Identify which features matter most for prediction
- Visualization: Feature importance plots

### Model Comparison
- Compare Random Forest vs Gradient Boosting vs SVM
- Cross-validation scores for robustness

---

## 10. Expected Outcomes

By the end of this project, you will have:

1. Unique ML Project - Not common among engineering graduates
2. Complete Dataset - 100-150 labeled Python repositories
3. Trained Model - Predicts code quality with reasonable accuracy (target: >75% test accuracy)
4. Publication-Ready Report - 20+ page LaTeX document
5. Reusable Tool - Code quality classifier that works on any Python repo
6. CV Impact - Demonstrates ML + software engineering expertise

---

## 11. Execution Instructions

### Starting Tomorrow:

**Day 1-3:** Phase 1 (Data Collection)
- Create scripts/01_scrape_github.py
- Collect high and low-quality repositories
- Create data/labels.csv

**Day 4-7:** Phase 2 (Feature Extraction)
- Create scripts/02_feature_extractor.py with CodeAnalyzer class
- Test on 5-10 sample repositories
- Generate data/extracted_features.csv

**Day 8-10:** Phase 3 (Data Preparation)
- Create scripts/03_data_preparation.py
- Clean, normalize, and split data

**Day 11-14:** Phase 4 (Model Training)
- Create scripts/04_train_model.py
- Train multiple models
- Evaluate and select best model

**Day 15-20:** Phase 5 (Report Writing)
- Document everything in LaTeX
- Create visualizations
- Write comprehensive report

---

## 12. Important Notes

- **One module at a time:** Complete each phase fully before moving to the next
- **Test incrementally:** Test feature extractor on 5 repos before scaling to 100
- **Use Colab for heavy computation:** MacBook M2 for development, Colab for training
- **Document as you go:** Keep notes on decisions, challenges, and solutions
- **No emoji:** Follow project guidelines

---

## 13. File Naming Convention

All Python scripts should follow this pattern:
```
01_scrape_github.py      # Phase 1
02_feature_extractor.py  # Phase 2
03_data_preparation.py   # Phase 3
04_train_model.py        # Phase 4
05_inference.py          # Bonus: Prediction on new repos
```

---

## 14. Quick Reference

**Start with Phase 1:**
- File: scripts/01_scrape_github.py
- Goal: Collect 50 high-quality and 50 low-quality repos
- Output: data/labels.csv

**Next: Phase 2:**
- File: scripts/02_feature_extractor.py
- Goal: Extract features from all repos
- Output: data/extracted_features.csv

---

**Status:** Ready to start
**Last Updated:** 2026-02-04
