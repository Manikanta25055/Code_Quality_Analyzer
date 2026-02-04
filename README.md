# Code Quality Prediction - ML Coursework Project

A unique machine learning project to predict Python code repository quality using static code analysis and machine learning.

---

## Project Summary

This project builds an ML classifier to determine whether a Python code repository is "high quality" or "low quality" WITHOUT executing the code. It combines software engineering knowledge with machine learning for a novel application.

### Key Features

1. **Unique Dataset:** Build your own labeled dataset of GitHub repositories
2. **AST-based Features:** Extract 25+ features using Python's Abstract Syntax Tree
3. **Multiple Models:** Compare Random Forest, Gradient Boosting, and SVM
4. **Comprehensive Report:** 20+ page LaTeX documentation
5. **Practical Application:** Real code quality detection tool

---

## Project Timeline

| Phase | Duration | Focus | Output |
|-------|----------|-------|--------|
| 1 | Week 1 | Data collection from GitHub | labels.csv (100 repos) |
| 2 | Week 2 | Feature extraction | extracted_features.csv |
| 3 | Week 2 | Data preparation | Normalized train/test sets |
| 4 | Week 3 | Model training | Trained classifier + metrics |
| 5 | Week 4 | Report writing | 20+ page PDF report |

**Start:** 2026-02-05
**Target Completion:** 4 weeks

---

## Feature Categories Extracted

### Structural Features
- Function count, average length
- Class size, nesting depth
- Cyclomatic complexity

### Naming Features
- Variable name quality
- Naming convention consistency
- Abbreviation ratio

### Code Pattern Features
- Error handling (try-except blocks)
- Type hints usage
- Docstring coverage
- Magic number count

### Maintainability Features
- Code duplication ratio
- Lines of code per file
- Module coupling

---

## Expected Model Performance

- **Target Accuracy:** 75%+
- **Target Precision/Recall:** 70%+
- **ROC-AUC:** 0.80+

---

## Files in This Project

```
code-quality-predictor/
├── README.md                       # This file
├── PROJECT_PLAN.md                 # Detailed implementation plan
├── SETUP.md                        # Setup instructions
├── requirements.txt                # Python dependencies
├── data/
│   ├── labels.csv                  # Repository labels (will create)
│   ├── extracted_features.csv      # Feature matrix (will create)
│   └── raw_repos/                  # Downloaded repositories
├── scripts/
│   ├── 01_scrape_github.py         # Data collection
│   ├── 02_feature_extractor.py     # Feature extraction
│   ├── 03_data_preparation.py      # Data cleaning
│   └── 04_train_model.py           # Model training
├── models/
│   └── quality_classifier.pkl      # Trained model
├── evaluation/
│   ├── metrics.txt                 # Performance metrics
│   └── confusion_matrix.png        # Visualizations
└── notebooks/
    └── exploratory_analysis.ipynb  # EDA on Google Colab
```

---

## Getting Started

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Review Documentation
Read PROJECT_PLAN.md to understand the complete project scope

### Step 3: Phase 1 - Data Collection (Tomorrow)
Start with scripts/01_scrape_github.py to collect repositories

---

## Dataset Definition

### High Quality Repositories (Label: 1)
- GitHub stars: 500+
- Recent commits (within 3 months)
- Has test directory
- Has comprehensive README
- Active maintenance

### Low Quality Repositories (Label: 0)
- GitHub stars: <50
- No commits in 1+ year
- No tests
- Minimal documentation
- Abandoned projects

---

## Technology Stack

**Language:** Python 3.8+
**ML Framework:** scikit-learn
**Data Processing:** pandas, numpy
**Visualization:** matplotlib, seaborn
**GitHub API:** PyGithub
**Code Analysis:** ast (built-in)

---

## Key Implementation Points

1. **Phase 1:** Use GitHub API to scrape 100 repositories
2. **Phase 2:** Extract 25+ features per repository using AST
3. **Phase 3:** Normalize features, handle missing values
4. **Phase 4:** Train 3 models, select best via cross-validation
5. **Phase 5:** Document results in comprehensive LaTeX report

---

## Important Notes

- Process one phase at a time
- Test on small dataset (5-10 repos) before scaling
- Use Google Colab for computationally intensive tasks
- Keep detailed notes of progress and challenges
- No emoji in documentation or outputs

---

## Success Criteria

- Successfully collect 100+ labeled repositories
- Extract 25+ meaningful features per repository
- Build model with 75%+ accuracy on test set
- Generate comprehensive 20+ page report
- Create reusable code quality prediction tool

---

## Resources

- **GitHub API Documentation:** https://docs.github.com/en/rest
- **Python AST Module:** https://docs.python.org/3/library/ast.html
- **scikit-learn Docs:** https://scikit-learn.org/
- **PyGithub Library:** https://pygithub.readthedocs.io/

---

## Next Actions

1. Install dependencies: `pip install -r requirements.txt`
2. Read PROJECT_PLAN.md carefully
3. Tomorrow: Begin Phase 1 (Data Collection)
4. Create scripts/01_scrape_github.py
5. Target: Collect first 10 high-quality repos as test

---

**Project Status:** Ready to Start
**Last Updated:** 2026-02-04
**Author:** Manikanta
**Course:** ML Coursework
