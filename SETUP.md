# Project Setup Guide

## Initial Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python3 -c "import pandas, numpy, sklearn, requests, github; print('All dependencies installed successfully!')"
```

### 3. Project Structure is Ready

Your project directory is now set up with:

```
code-quality-predictor/
├── data/
│   └── raw_repos/          # Will store downloaded repos
├── scripts/                # Where you'll write Python scripts
├── models/                 # For storing trained models
├── evaluation/             # For storing results and visualizations
├── notebooks/              # For Jupyter/Colab notebooks
├── PROJECT_PLAN.md         # Complete project documentation
├── SETUP.md                # This file
└── requirements.txt        # Python dependencies
```

---

## Tomorrow's Starting Point

### Phase 1: Data Collection

**Script:** scripts/01_scrape_github.py

**Goal:** Collect 100 repositories (50 high-quality, 50 low-quality)

**Steps:**
1. Use GitHub API to find repos with specific criteria
2. High-quality: 500+ stars, recent commits
3. Low-quality: <50 stars, abandoned
4. Save repository names and labels to data/labels.csv

**Expected Output:**
- data/labels.csv with columns: [repo_url, repo_name, quality_label]

---

## Important Notes

- Work on one phase at a time
- Test with small datasets first (5-10 repos)
- Use Google Colab for heavy computation if needed
- Keep detailed notes of your progress
- Run tests after each major component

---

## Quick Commands

### Check current directory
```bash
pwd
cd code-quality-predictor
```

### Create a new Python script
```bash
touch scripts/01_scrape_github.py
```

### Run a Python script
```bash
python3 scripts/01_scrape_github.py
```

### List all files in project
```bash
find . -type f -name "*.py" -o -name "*.csv" -o -name "*.pkl"
```

---

## Next Steps

1. Read PROJECT_PLAN.md completely
2. Understand Phase 1 requirements
3. Tomorrow: Start writing 01_scrape_github.py
4. Collect first 10 high-quality repositories as test

---

**Ready to Start:** Yes
**Start Date:** Tomorrow (2026-02-05)
