# Project Checklist - Code Quality Prediction ML

## Pre-Project Setup

- [x] Project directory structure created
- [x] Documentation files created (README.md, PROJECT_PLAN.md, SETUP.md)
- [x] requirements.txt file created with all dependencies
- [x] data/, scripts/, models/, evaluation/, notebooks/ directories created
- [x] Project location: /Users/manikantagonugondla/Desktop/MIT/MIT/3rd Year/6th sem/ML/code-quality-predictor

---

## Phase 1: Data Collection (Week 1)

### Before Starting
- [ ] Read README.md completely
- [ ] Read PROJECT_PLAN.md section 1-2
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify installation: `python3 -c "import pandas, numpy, sklearn; print('OK')"`

### Implementation
- [ ] Create scripts/01_scrape_github.py
- [ ] Implement function to search GitHub API for high-quality repos
- [ ] Implement function to search GitHub API for low-quality repos
- [ ] Set up error handling and logging
- [ ] Test with 10 repos first (5 high, 5 low)
- [ ] Scale to 100 repos (50 high, 50 low)

### Testing
- [ ] Verify data/labels.csv is created
- [ ] Check CSV format: [repo_url, repo_name, quality_label]
- [ ] Verify labels are correct (0 or 1)
- [ ] Count: 50 high-quality (label=1), 50 low-quality (label=0)

### Deliverable
- [ ] data/labels.csv with 100 repositories and labels

---

## Phase 2: Feature Extraction (Week 2)

### Before Starting
- [ ] Review PROJECT_PLAN.md section 3-5
- [ ] Study Python AST module documentation
- [ ] Understand all 25+ features to extract

### Implementation
- [ ] Create scripts/02_feature_extractor.py
- [ ] Create CodeAnalyzer class
- [ ] Implement structural feature extraction
- [ ] Implement naming convention feature extraction
- [ ] Implement code pattern feature extraction
- [ ] Implement maintainability feature extraction
- [ ] Add error handling for syntax errors
- [ ] Test on 5 sample repositories

### Testing
- [ ] Verify features are extracted correctly
- [ ] Check for missing values (NaN)
- [ ] Verify feature ranges are reasonable
- [ ] Compare features across high vs low quality repos

### Deliverable
- [ ] data/extracted_features.csv with 100 rows and 25+ columns

---

## Phase 3: Data Preparation (Week 2-3)

### Before Starting
- [ ] Review PROJECT_PLAN.md section 6
- [ ] Study sklearn preprocessing module

### Implementation
- [ ] Create scripts/03_data_preparation.py
- [ ] Handle missing values (fill with mean or 0)
- [ ] Standardize/normalize features using StandardScaler
- [ ] Create train-test split (80-20)
- [ ] Check for class balance
- [ ] Visualize feature distributions

### Testing
- [ ] Verify normalized features are in similar ranges
- [ ] Check train set: 80 repos (40 high, 40 low)
- [ ] Check test set: 20 repos (10 high, 10 low)
- [ ] Visualize class distribution

### Deliverable
- [ ] Cleaned, normalized train and test sets ready for ML

---

## Phase 4: Model Training & Evaluation (Week 3-4)

### Before Starting
- [ ] Review PROJECT_PLAN.md section 7-8
- [ ] Study scikit-learn classification models

### Implementation
- [ ] Create scripts/04_train_model.py
- [ ] Implement Random Forest classifier
- [ ] Implement Gradient Boosting classifier
- [ ] Implement SVM classifier
- [ ] Perform cross-validation on each model
- [ ] Select best model based on CV score
- [ ] Train best model on full training set
- [ ] Evaluate on test set

### Testing
- [ ] Print accuracy, precision, recall, F1-score
- [ ] Generate confusion matrix
- [ ] Plot ROC-AUC curve
- [ ] Compare all three models

### Metrics to Track
- [ ] Train accuracy: ___
- [ ] Test accuracy: ___ (target: 75%+)
- [ ] Precision: ___ (target: 70%+)
- [ ] Recall: ___
- [ ] F1-Score: ___
- [ ] ROC-AUC: ___ (target: 0.80+)

### Deliverable
- [ ] models/quality_classifier.pkl (trained model)
- [ ] evaluation/metrics.txt (all metrics)
- [ ] evaluation/confusion_matrix.png (visualization)
- [ ] evaluation/roc_auc_curve.png (visualization)

---

## Phase 5: Report Writing (Week 4)

### Report Contents (20+ pages, Times New Roman 12pt)
- [ ] Title page and abstract
- [ ] Table of contents
- [ ] 1. Introduction & Motivation
- [ ] 2. Literature Review
- [ ] 3. Methodology
  - [ ] 3.1 Feature Engineering
  - [ ] 3.2 Dataset Collection
  - [ ] 3.3 Data Preparation
  - [ ] 3.4 Model Selection
- [ ] 4. Experimental Setup
  - [ ] 4.1 Dataset Details
  - [ ] 4.2 Model Hyperparameters
  - [ ] 4.3 Evaluation Metrics
- [ ] 5. Results & Analysis
  - [ ] 5.1 Model Performance
  - [ ] 5.2 Feature Importance
  - [ ] 5.3 Confusion Matrix & ROC Curve
  - [ ] 5.4 Model Comparison
- [ ] 6. Discussion
- [ ] 7. Conclusion & Future Work
- [ ] 8. References
- [ ] Appendix A: Code Snippets

### Deliverable
- [ ] report.pdf (20+ pages, well-formatted)

---

## Final Verification

- [ ] All 5 phases completed
- [ ] All deliverables present
- [ ] Project documentation complete
- [ ] Code is well-commented
- [ ] Results are reproducible
- [ ] Report is comprehensive

---

## Project Statistics

**Total Lines of Code:** ___
**Total Features Extracted:** ___
**Datasets Used:** 100 GitHub repositories
**Models Trained:** 3 (Random Forest, Gradient Boosting, SVM)
**Best Model Accuracy:** ___%
**Report Pages:** ___

---

## Key Achievements

- [ ] Unique ML project (not common)
- [ ] Custom dataset created (100+ repos)
- [ ] 25+ hand-crafted features
- [ ] ML model with 75%+ accuracy
- [ ] Comprehensive documentation
- [ ] Practical code quality tool

---

## Sign-Off

- [ ] Project completed on time
- [ ] All requirements met
- [ ] Documentation complete
- [ ] Ready for submission

---

**Project Start Date:** 2026-02-05
**Expected Completion:** 4 weeks
**Project Status:** Ready to Begin

Remember: One phase at a time. Test incrementally. Document as you go.
