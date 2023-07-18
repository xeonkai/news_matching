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

### To run app
```bash
streamlit run app/topic-suggestion-interface-v3/Home.py
```

### To add additional libraries, and update environment

1. Add the core libaries to the requirements-base.txt
```
streamlit
pandas<2 # st-aggrid requirement
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
pip freeze > requirements.txt
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

Build container
```bash
docker build -t <repo_name>/news-matching:latest .
docker tag <repo_name>/news-matching:latest <repo_name>/news-matching:latest
docker push <repo_name>/news-matching:latest
```

Run container
```bash
docker compose up -d
echo http://localhost:8501/
# Do whatever
...
docker compose down
```

Build & push container
```bash
docker build -t news-matching:latest .
docker tag news-matching:latest <repo>/news-matching:latest
docker push <repo>/news-matching:latest
```

```bash
docker compose pull
docker compose up -d
```