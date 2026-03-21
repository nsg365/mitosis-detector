# GitHub Setup Guide

This document provides step-by-step instructions for publishing your mitosis detection project to GitHub.

## Prerequisites

- GitHub account (https://github.com)
- Git installed locally
- SSH key configured (recommended) or HTTPS token ready

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. **Repository name:** `Mitosis-Detection-TUPAC16`
3. **Description:** "Two-stage deep learning pipeline for mitosis detection in histopathology images"
4. **Visibility:** Public
5. **Initialize:** Leave unchecked (you'll push existing code)
6. Click **Create repository**

## Step 2: Prepare Local Repository

```bash
cd /Users/nihar/Downloads/College/6th\ Sem/NNDL/CourseProject

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Two-stage mitosis detection pipeline"
```

## Step 3: Connect to GitHub

```bash
# Add remote (replace with your username)
git remote add origin https://github.com/YourUsername/Mitosis-Detection-TUPAC16.git

# Verify
git remote -v
```

## Step 4: Push to GitHub

```bash
# Set up tracking and push
git branch -M main
git push -u origin main
```

## Step 5: Organize for Publication

### Clean Up Generated Files

Before pushing, remove large files not needed on GitHub:

```bash
# Remove checkpoints (add to .gitignore, upload separately if needed)
git rm --cached checkpoints/*.pth

# Remove large datasets
git rm --cached data/raw/*
git rm --cached data/processed/*

# Remove output logs
git rm --cached outputs/*.log

# Commit cleanup
git commit -m "Remove large files from repo"
```

### Add README to Root

✅ Already created: `README.md` with quick start guide

### Add License

✅ Already created: `LICENSE` (MIT)

### Add .gitignore

✅ Already created: `.gitignore` with standard patterns

## Step 6: Create Additional Documentation

### GitHub Workflow (Optional)

Create `.github/workflows/tests.yml` for CI/CD:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Verify imports
      run: python -c "import torch; import models; import configs"
```

### Topics/Tags on GitHub

Add these topics to improve discoverability:
- `deep-learning`
- `object-detection`
- `histopathology`
- `medical-imaging`
- `pytorch`
- `faster-rcnn`
- `computer-vision`

## Step 7: Create Releases

Tag your submission:

```bash
git tag -a v1.0 -m "Final course project submission"
git push origin v1.0
```

## Step 8: Set Up GitHub Pages (Optional)

To make report accessible online:

1. Go to repository Settings → Pages
2. Source: `main` branch
3. Create `docs/` folder and add REPORT.pdf there
4. Your report will be available at: `https://yourusername.github.io/Mitosis-Detection-TUPAC16/`

## Directory Structure for GitHub

Your repository should look like:

```
.
├── .gitignore                    ✅ Excludes large files
├── .github/
│   └── workflows/
│       └── tests.yml             (optional)
├── LICENSE                       ✅ MIT License
├── README.md                     ✅ Main documentation
├── ABSTRACT.md                   ✅ 5-sentence abstract
├── CITATIONS.md                  ✅ Citation formats
├── REPORT.tex                    ✅ Full technical report
├── requirements.txt              ✅ Dependencies
│
├── models/                       ✅ Clean modules
│   ├── __init__.py
│   ├── stage1_classifier.py
│   └── stage2_detector.py
│
├── configs/                      ✅ Configuration
│   └── __init__.py
│
├── src/                          ✅ Pipeline scripts
│   ├── preprocess.py
│   ├── pipeline.py
│   ├── evaluate.py
│   ├── stage1_classifier.py
│   └── stage2_detector.py
│
├── gradio_demo.py                ✅ Interactive UI
│
├── outputs/                      ⚠️  (exclude large files, add summary)
│   ├── cross_center_results.csv
│   └── README.md                 (describe what's here)
│
├── docs/                         (optional)
│   └── REPORT.pdf
│
└── data/                         ⚠️  (git-ignored)
    ├── raw/
    └── processed/
```

## Step 9: Update Repository on GitHub

Make repository discoverable:

1. **Description:** "Two-stage deep learning for mitosis detection in histopathology"
2. **URL:** (if deploying): https://yourusername.github.io/Mitosis-Detection-TUPAC16/
3. **Topics:** Add the tags above
4. **Pinned README:** Check "Include in the home include"

## Step 10: Final Checklist

```
✅ Modular code in src/ and models/
✅ Clean configurations in configs/
✅ License file (MIT)
✅ .gitignore properly configured
✅ README.md with quick start
✅ ABSTRACT.md (5 sentences max)
✅ CITATIONS.md with IEEE/NeurIPS formats
✅ REPORT.tex (comprehensive technical report)
✅ requirements.txt updated
✅ Large checkpoints excluded from git
✅ Dataset excluded from git
✅ Interactive demo (gradio_demo.py)
✅ GitHub repository public and documented
✅ Repository tagged with v1.0 release
```

## Usage Instructions for Others

When someone clones your repo:

```bash
git clone https://github.com/YourUsername/Mitosis-Detection-TUPAC16.git
cd Mitosis-Detection-TUPAC16

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Follow README.md for next steps
```

## Publishing to arXiv (Optional)

To submit as a technical report to arXiv:

1. Convert REPORT.tex to PDF:
   ```bash
   pdflatex REPORT.tex
   ```

2. Go to https://arxiv.org/submit
3. Upload PDF and supplementary materials
4. Fill metadata (title, authors, abstract)
5. Submit

## Troubleshooting

### Large file warning
```bash
git rm --cached path/to/large/file
echo "*.pth" >> .gitignore
git commit -am "Remove large checkpoint files"
```

### Need to update after pushing
```bash
git add .
git commit -m "Update: [description]"
git push origin main
```

### Fix commit message
```bash
git commit --amend -m "New message"
git push origin main --force-with-lease
```

## Contact & Support

- For GitHub help: https://docs.github.com/
- For Git tutorials: https://git-scm.com/
- For research publishing: https://arxiv.org/

---

**Last Updated:** March 21, 2026
