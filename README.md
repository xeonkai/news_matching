# News Matching

## Folder structure
```bash
.
├──app/
│   └── pages/
├──data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── figures/
├── reports/
├── results/
├── scripts/
│   └──deprecated/
├── .gitignore
├── environment.yml
├── README.md
└── requirements.txt
```

## Setup instructions

### To reproduce and use environment with required libaries
Takes like 10mins for first time ~
```bash
# Create environment from file
conda env create -f environment.yml

# To use created environment
conda activate news_matching

# Open jupyter lab and code
jupyter lab

# Ctrl + C to stop jupyter server

# Deactivate environment / close terminal
conda deactivate
```

### To setup folder structure and prepare embeddings, and run app
Run notebooks 01, 02, 03
```bash
jupyter lab

# Run notebooks 01, 02, 03 in order

streamlit run app/main.py
```

### To add additional libraries, and update environment

1. Add the relevant libaries to the environment.yml (Can omit versions until we have a stable library base)
```yaml
name: news_matching
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - python-dotenv
  ...
  - <YOUR LIBRARY NAME>
  ...
  - pip:
      - -r requirements.txt
```

2. Update existing environment
```bash
conda env update -f environment.yml

# Restart jupyter lab
```

3. Export environment config with versions
```bash
conda env export --no-builds > environment.lock.yml
```

4. Cleanup when project is completed
```bash
conda env remove –n news_matching

conda clean --all
```