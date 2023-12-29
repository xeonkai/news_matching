# News Matching

## Folder structure
```bash
.
├── app/
│   ├── utils/
│   └── pages/
├── scripts/
├── data/
│   ├── metrics/
│   ├── raw_upload/
│   ├── taxonomy/
│   └── train/
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup instructions

### To reproduce and use environment with required libaries
Takes like 10mins for first time ~
```bash
# Create virtual env
python -m venv venv
# To use created environment
source venv/bin/activate
# To install packages from file
python -m pip install -r requirements.txt
# Do stuff
...
# Deactivate environment / close terminal when done
deactivate
```

### To train default model
```bash
# requires "all_tagged_articles_new.csv" in root folder
python scripts/modelling.py
```

### To run app
```bash
# requires "default model" and "all_tagged_articles_new.csv"
streamlit run app/main.py
```

### To add additional libraries, and update environment

1. Add the core libaries to the requirements-base.txt
```
streamlit
numpy
scikit-learn
...
...
<YOUR LIBRARY NAME>
```

2. Update & export new environment
```bash
# Exit old environment
deactivate
# Purge old environment
rm -rf venv
# Create fresh environmnet
python -m venv venv
# Activate environment
source venv/bin/activate
# Install package with auto-resolver
pip install -r requirements-base.txt
# Export locked versions for team use
echo "--extra-index-url https://download.pytorch.org/" > text.txt
pip freeze >> requirements.txt
# Make sure app is working
...
```

### Build & run on docker

Install docker if not available

```bash
# Install docker desktop for windows / mac
https://www.docker.com/products/docker-desktop/

# For linux server install
sudo apt-get update && sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo groupadd docker
sudo usermod -aG docker $USER
# Reboot
# Linux also need to explicit install compose plugin
sudo apt-get install docker-compose-plugin
```

Build & push container
```bash
docker buildx build --push --platform linux/arm64/v8,linux/amd64 --tag <repo_name>/news-matching:latest .
```

Run container from docker hub
```bash
docker compose up -d
echo http://localhost:8501/
# Do whatever
...
docker compose down
```