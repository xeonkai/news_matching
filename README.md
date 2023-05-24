# News Matching

## Folder structure
```bash
.
├──topic-suggestion-interface-v3/
│   └── pages/
├──data/
│   ├── raw/
│   └── processed/
├── .gitignore
├── environment.yml
├── README.md
├── environment.yml
└── requirements.txt
```

## Setup instructions

### To reproduce and use environment with required libaries
Takes like 10mins for first time ~
```bash
# # Create environment from file
# conda env create -f environment.yml
# # To use created environment
# conda activate news_matching
# # Deactivate environment / close terminal
# conda deactivate

# Create virtual env
python -m venv venv
# To use created environment
source venv/bin/activate
# To install packages from file
python -m pip install -r requirements.txt
# Deactivate environment / close terminal
deactivate
```

### To run app
```bash
streamlit run app/topic-suggestion-interface-v3.py
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