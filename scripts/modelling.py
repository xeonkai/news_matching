import datetime
import json
import os
import shutil
from pathlib import Path

import duckdb
import gcsfs
import pandas as pd
from datasets import Features, Value, load_dataset
from dotenv import load_dotenv
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report, top_k_accuracy_score

load_dotenv()
GSHEET_TAXONOMY_ID = os.environ.get("GSHEET_TAXONOMY_ID")
gsheet_taxonomy_url = "https://docs.google.com/spreadsheets/d/" + GSHEET_TAXONOMY_ID

GCS_BUCKET = os.environ["GCS_BUCKET"]
service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
fs = gcsfs.GCSFileSystem(
    project=service_account_info["project_id"], token=service_account_info
)


def prepare_base_training_dataset():
    DATA_DIR = Path("data")
    (DATA_DIR / "train").mkdir(parents=True, exist_ok=True)

    pd.read_csv(
        "all_tagged_articles_new.csv",
        usecols=["Published", "Headline", "Theme", "New Index"],
        na_values="-",
        parse_dates=["Published"],
    ).rename(
        lambda col_name: col_name.lower().replace(" ", "_"), axis="columns"
    ).assign(
        label=lambda df: df[["theme", "new_index"]]
        .fillna("")
        .agg(" > ".join, axis="columns")
    ).drop(
        columns=["theme", "new_index"]
    ).to_parquet(
        DATA_DIR / "train" / "base_training_data.parquet"
    )


def create_dataset(min_labels=2):
    DATA_DIR = Path("data")
    if not (DATA_DIR / "train" / "base_training_data.parquet").exists():
        prepare_base_training_dataset()

    with duckdb.connect(str(DATA_DIR / "news.db")) as con:
        con.sql(
            f"""
            COPY 
            (SELECT published, headline, label FROM daily_news WHERE label IS NOT NULL) 
            TO '{DATA_DIR / "train"}/generated_training_data.parquet'
            (FORMAT PARQUET);
            """
        )

    features = Features(
        {
            "published": Value("timestamp[ns]"),
            "headline": Value("string"),
            "label": Value("string"),
        }
    )

    min_labels_list = (
        pd.concat(
            pd.read_parquet(parquet_file, columns=["label"])
            for parquet_file in (DATA_DIR / "train").glob("*.parquet")
        )["label"]
        .value_counts()[lambda s: s >= min_labels]
        .index.to_list()
    )

    dataset = load_dataset(
        "parquet", data_dir=str(DATA_DIR / "train"), features=features
    ).filter(lambda row: row["label"] in min_labels_list)

    return dataset


def train_test_split_eval_model(top_k=20):
    dataset = create_dataset(min_labels=10)
    # Evaluation split
    temp_dataset = (
        dataset["train"]
        .add_column("label_class", dataset["train"]["label"])
        .class_encode_column("label_class")
        .train_test_split(test_size=0.2, stratify_by_column="label_class")
    )
    train_dataset = temp_dataset["train"]
    eval_dataset = temp_dataset["test"]

    # train_dataset = dataset["train"]

    # Load a SetFit model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        cache_dir="cached_models",
    )

    # Create trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=20,  # The number of text pairs to generate for contrastive learning
        num_epochs=1,  # The number of epochs to use for contrastive learning
        column_mapping={
            "headline": "text",
            "label": "label",
        },  # Map dataset columns to text/label expected by trainer
    )

    trainer.train()

    y_true = eval_dataset["label"]
    y_score = trainer.model.predict_proba(eval_dataset["headline"])
    y_pred = trainer.model.predict(eval_dataset["headline"])

    k_range = list(range(1, top_k))
    top_k_scores = [
        top_k_accuracy_score(
            y_true, y_score, k=k, labels=trainer.model.model_head.classes_
        )
        for k in k_range
    ]

    top_k_df = pd.DataFrame({"k": k_range, "accuracy": top_k_scores})
    class_scores_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=trainer.model.model_head.classes_,
            output_dict=True,
        )
    ).T

    top_k_df.to_csv(
        f"gs://{GCS_BUCKET}/metrics/top_k_scores_{datetime.date.today().isoformat()}.csv",
        index=False,
    )
    class_scores_df.to_csv(
        f"gs://{GCS_BUCKET}/metrics/class_scores_{datetime.date.today().isoformat()}.csv",
        index=False,
    )


def train_model(model_path=None):
    dataset = create_dataset()
    train_dataset = dataset["train"]

    # Load a SetFit model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        cache_dir="cached_models",
    )

    # Create trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=20,  # The number of text pairs to generate for contrastive learning
        num_epochs=1,  # The number of epochs to use for contrastive learning
        column_mapping={
            "headline": "text",
            "label": "label",
        },  # Map dataset columns to text/label expected by trainer
    )

    if model_path is None:
        model_path = f"trained_models/{datetime.date.today().isoformat()}"

    trainer.train()
    trainer.model.save_pretrained(model_path)
    fs.put(model_path, f"gs://{GCS_BUCKET}/{model_path}", recursive=True)


def delete_old_models(models_to_keep=24):
    # Keep 24 most recent trained models locally
    data_folder = Path("trained_models")
    data_folder_date_sorted = sorted(data_folder.glob("20*"))
    for old_model_path in data_folder_date_sorted[:-models_to_keep]:
        shutil.rmtree(old_model_path)


if __name__ == "__main__":
    # Generate default model
    train_model()
    # train_test_split_eval_model()
    delete_old_models()
