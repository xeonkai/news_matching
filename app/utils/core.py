import datetime
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"


def fetch_taxonomy() -> pd.DataFrame:
    taxonomy_folder = DATA_DIR / "taxonomy"
    taxonomy_folder.mkdir(parents=True, exist_ok=True)
    taxonomy_date_sorted = sorted(taxonomy_folder.glob("20*"), key=os.path.getmtime)

    if len(taxonomy_date_sorted) == 0:
        taxonomy_df = (
            pd.read_csv("all_tagged_articles_new.csv", usecols=["Theme", "New Index"])
            .rename(columns={"New Index": "Index"})
            .drop_duplicates()
            .sort_values(["Theme", "Index"])
            .reset_index(drop=True)
        )
    else:
        taxonomy_df = pd.read_parquet(taxonomy_date_sorted[-1])

    return taxonomy_df


def save_taxonomy(df):
    taxonomy_folder = DATA_DIR / "taxonomy"
    (
        df.drop_duplicates()
        .sort_values(["Theme", "Index"])
        .reset_index(drop=True)
        # drop incomplete rows
        .to_parquet(taxonomy_folder / datetime.date.today().isoformat())
    )


def load_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer(
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        cache_folder="cached_models",
    )
    return model


def load_classification_model(model_path=None) -> SetFitModel:
    data_folder = Path("trained_models")
    data_folder_date_sorted = sorted(data_folder.glob("20*"), key=os.path.getmtime)
    if len(data_folder_date_sorted) == 0:
        latest_model_path = "default_model"
    else:
        latest_model_path = str(data_folder_date_sorted[-1])

    if model_path is None:
        model_path = latest_model_path

    model = SetFitModel.from_pretrained(model_path)
    return model


def label_df(df: pd.DataFrame, model: SetFitModel, column: str) -> pd.DataFrame:
    y_score = model.predict_proba(df[column])

    label_order = np.argsort(y_score, axis=1, kind="stable").numpy()[:, ::-1]
    label_scores_df = pd.DataFrame(y_score, columns=model.model_head.classes_)

    sorted_label_list = []
    sorted_scores_list = []
    for idx, row in label_scores_df.iterrows():
        sorted_label = row.iloc[label_order[idx]]
        sorted_label_list.append(sorted_label.index.to_list())
        sorted_scores_list.append(sorted_label.to_list())

    labelled_df = df.assign(
        predicted_indexes=sorted_label_list, prediction_prob=sorted_scores_list
    )

    labelled_df = df.assign(
        suggested_labels=sorted_label_list,
        suggested_labels_score=sorted_scores_list,
    )
    return labelled_df


def embed_df(df: pd.DataFrame, model: SentenceTransformer, column: str) -> pd.DataFrame:
    embedded_df = df.assign(vector=model.encode(df[column]).tolist())
    return embedded_df


class FileHandler:
    DAILY_NEWS_TABLE = "daily_news"

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.db_path = str(self.data_dir / "news.db")

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        with duckdb.connect(self.db_path) as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                {self.DAILY_NEWS_TABLE}(
                    "published" TIMESTAMP,
                    "headline" VARCHAR,
                    "summary" VARCHAR,
                    "link" VARCHAR UNIQUE,
                    "domain" VARCHAR,
                    "facebook_interactions" BIGINT,  
                    "source" VARCHAR,
                    "label" VARCHAR
                );
                """
            )

    @staticmethod
    def preprocess_daily_scan(file, source: str = "") -> pd.DataFrame:
        df = (
            pd.read_csv(
                file,
                usecols=[
                    "Published",
                    "Headline",
                    "Summary",
                    "Link",
                    "Domain",
                    "Facebook Interactions",
                ],
                dtype={
                    "Headline": "string",
                    "Summary": "string",
                    "Link": "string",
                    "Domain": "string",
                    "Facebook Interactions": "int",
                },
                parse_dates=["Published"],
            )
            .assign(source=source)
            .rename(lambda col_name: col_name.lower().replace(" ", "_"), axis="columns")
        )
        return df

    def write_csv(self, file):
        filepath = self.raw_data_dir / file.name
        filepath.write_bytes(file.getbuffer())
        return filepath

    def write_db(self, file) -> None:
        processed_table = self.preprocess_daily_scan(file, source=file.name)
        with duckdb.connect(self.db_path) as con:
            #
            con.sql(f"DELETE FROM {self.DAILY_NEWS_TABLE} WHERE source = '{file.name}'")
            # Append to table, replace if existing link found
            con.sql(
                f"""
                INSERT INTO {self.DAILY_NEWS_TABLE} 
                SELECT published, headline, summary, link, domain, facebook_interactions, source, NULL 
                FROM processed_table
                ON CONFLICT
                DO UPDATE
                    SET published = published,
                    headline = headline,
                    summary = summary,
                    domain = domain,
                    facebook_interactions = facebook_interactions,
                    source = source
                """
            )

    def update_labels(self, df):
        with duckdb.connect(self.db_path) as con:
            con.sql(
                f"""
                UPDATE {self.DAILY_NEWS_TABLE} 
                SET label = df.label 
                FROM df 
                WHERE {self.DAILY_NEWS_TABLE}.link = df.link;
                """
            )

    def get_filter_bounds(self):
        with duckdb.connect(self.db_path) as con:
            max_fb_interactions = con.sql(
                f"SELECT MAX(facebook_interactions) FROM {self.DAILY_NEWS_TABLE}"
            ).fetchone()[0]

            domain_list = [
                domain
                for domain, count in con.sql(
                    "SELECT domain, COUNT(*) AS count FROM "
                    f"{self.DAILY_NEWS_TABLE} "
                    "GROUP BY domain "
                    "ORDER BY count DESC"
                ).fetchall()
            ]

            min_date, max_date = con.sql(
                f"SELECT MIN(published), MAX(published) FROM "
                f"{self.DAILY_NEWS_TABLE} "
            ).fetchone()
        return {
            "max_fb_interactions": max_fb_interactions,
            "domain_list": domain_list,
            "min_date": min_date,
            "max_date": max_date,
        }

    def query(self, query):
        with duckdb.connect(self.db_path) as con:
            results_df = con.sql(query).to_df()
        return results_df

    def filtered_query(self, columns, domain_filter, min_engagement, date_range):
        query = (
            f"SELECT {','.join(columns)} "
            f"FROM {self.DAILY_NEWS_TABLE} "
            f"WHERE domain NOT IN {tuple(domain_filter) if domain_filter else ('NULL',)} "
            f"AND facebook_interactions >= {min_engagement} "
            f"AND published BETWEEN '{date_range[0]}' AND '{date_range[1]}' "
            f"ORDER BY facebook_interactions DESC   "
            # f"{f'LIMIT {limit}' if limit else ''} "
        )
        with duckdb.connect(self.db_path) as con:
            results_filtered = con.sql(query).to_df()
        return results_filtered

    def list_csv_filenames(self):
        return [file.name for file in self.raw_data_dir.iterdir()]

    def list_csv_files(self):
        return list(self.raw_data_dir.iterdir())

    def list_csv_files_df(self):
        raw_files_info = [
            {
                "filename": file.name,
                "modified": datetime.datetime.fromtimestamp(file.stat().st_mtime),
                # "filesize": f"{file.stat().st_size / 1000 / 1000:.2f} MB",
            }
            for file in self.raw_data_dir.iterdir()
        ]
        return pd.DataFrame(raw_files_info)

    def remove_files(self, filenames):
        for filename in filenames:
            (self.raw_data_dir / filename).unlink(missing_ok=True)
            with duckdb.connect(self.db_path) as con:
                con.sql(
                    f"DELETE FROM {self.DAILY_NEWS_TABLE} WHERE source = '{filename}'"
                )
        return [self.raw_data_dir / filename for filename in filenames]

    def download_csv_files(self):
        # Convert folder to zip, return zip file
        raise NotImplementedError

    def __repr__(self):
        return f"FileHandler({repr(self.data_dir)})"

    def __str__(self):
        return f"{[file.name for file in self.raw_data_dir.iterdir()]}"

    def __len__(self):
        return len(self.raw_data_dir.iterdir())
