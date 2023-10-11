import datetime
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel
from datetime import datetime, timedelta

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
WEEKLY_DATA_DIR = DATA_DIR / "weekly"


def fetch_default_taxonomy() -> pd.DataFrame:
    return (
        pd.read_csv("all_tagged_articles_new.csv", usecols=["Theme", "New Index"])
        .rename(columns={"New Index": "Index"})
        .drop_duplicates()
        .sort_values(["Theme", "Index"])
        .reset_index(drop=True)
    )


def list_taxonomies() -> list[Path]:
    taxonomy_folder = DATA_DIR / "taxonomy"
    taxonomy_folder.mkdir(parents=True, exist_ok=True)
    taxonomy_date_sorted = sorted(taxonomy_folder.glob("20*"), key=os.path.getmtime)
    taxonomy_date_sorted.append("Default")
    return taxonomy_date_sorted


def fetch_taxonomy(path: Path) -> pd.DataFrame:
    if str(path) == "Default":
        return fetch_default_taxonomy()
    return pd.read_parquet(path)


def fetch_latest_taxonomy() -> pd.DataFrame:
    taxonomy_date_sorted = list_taxonomies()

    if len(taxonomy_date_sorted) == 0:
        taxonomy_df = fetch_default_taxonomy()
    else:
        taxonomy_df = fetch_taxonomy(taxonomy_date_sorted[-1])

    return taxonomy_df


def save_taxonomy(df):
    taxonomy_folder = DATA_DIR / "taxonomy"
    (
        df.drop_duplicates()
        .sort_values(["Theme", "Index"])
        .reset_index(drop=True)
        # TODO: drop incomplete rows
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
                    "link" VARCHAR,
                    "domain" VARCHAR,
                    "facebook_interactions" BIGINT,  
                    "source" VARCHAR,
                    "label" VARCHAR,
                    PRIMARY KEY ("published", "link")
                );
                """
            )

    @staticmethod
    def preprocess_daily_scan(file, source: str = "") -> pd.DataFrame:
        df = (
            pd.read_excel(
                file,
                usecols=[
                    "Published",
                    "Headline",
                    "Summary",
                    "Link URL",
                    "Domain",
                    "Facebook Interactions",
                ],
                dtype={
                    "Published": "string",
                    "Headline": "string",
                    "Summary": "string",
                    "Link URL": "string",
                    "Domain": "string",
                    "Facebook Interactions": "int",
                },
                # parse_dates=["Published"],
            )
            .assign(source=source)
            .rename(lambda col_name: col_name.lower().replace(" ", "_"), axis="columns")
            .rename(columns={"link_url": "link"})
        )

        df["published"] = pd.to_datetime(df['published'])
        df = df.dropna(subset=['link'])
        data_name = file.name
        date_string = data_name.partition("posts-")[2].partition(" -")[0]
        format = '%m_%d_%y-%H_%M'
        latest_date_file = datetime.strptime(date_string, format).date()

        return df, latest_date_file

    @staticmethod
    def preprocess_labelled_articles(file, source: str = "") -> pd.DataFrame:
        df = (
            pd.read_excel(
                file,
                usecols=[
                    "published",
                    "headline",
                    "summary",
                    "link",
                    "domain",
                    "facebook_interactions",
                    "label",
                    "source"
                ],
                dtype={
                    "published": "string",
                    "headline": "string",
                    "summary": "string",
                    "link": "string",
                    "domain": "string",
                    "facebook_interactions": "int",
                    "label": "string",
                    "source": "string"
                },
            )
        )
        return df

    def write_csv(self, file):
        filepath = self.raw_data_dir / file.name
        filepath.write_bytes(file.getbuffer())
        return filepath

    def write_db(self, file) -> None:
        processed_table, latest_date_file = self.preprocess_daily_scan(file, source=file.name)

        query = (
            f"SELECT max(published) max_published "
            f"FROM {self.DAILY_NEWS_TABLE} "
        )
        with duckdb.connect(self.db_path) as con:
            results_filtered = con.sql(query).to_df()

        latest_date_db = results_filtered['max_published'].dt.date[0]
        if pd.isnull(latest_date_db):
            insert_table = processed_table
        else:
            insert_table = processed_table.loc[processed_table['published'].dt.date.between(latest_date_db, latest_date_file, inclusive=False)]
        #
        with duckdb.connect(self.db_path) as con:
            #
            con.sql(f"DELETE FROM {self.DAILY_NEWS_TABLE} WHERE source = '{file.name}'")
            # Append to table, replace if existing link found
            con.sql(
                f"""
                INSERT INTO {self.DAILY_NEWS_TABLE}
                SELECT published, headline, summary, link, domain, facebook_interactions, source, NULL
                FROM insert_table
                ON CONFLICT (published, link)
                DO UPDATE
                    SET headline = headline,
                    summary = summary,
                    domain = domain,
                    facebook_interactions = facebook_interactions,
                    source = source
                """
            )

    def write_labelled_articles_db(self, file) -> None:
        processed_table = self.preprocess_labelled_articles(file)
        min_date = min(processed_table['published'])
        max_date = max(processed_table['published'])
        with duckdb.connect(self.db_path) as con:

            con.sql(f"DELETE FROM {self.DAILY_NEWS_TABLE} WHERE published >= '{min_date}' AND published <= '{max_date}'")
            # Append to table, replace if existing link found
            con.sql(
                f"""
                INSERT INTO {self.DAILY_NEWS_TABLE} 
                SELECT published, headline, summary, link, domain, facebook_interactions, source, label 
                FROM processed_table
                ON CONFLICT (published, link)
                DO UPDATE
                    SET headline = headline,
                    summary = summary,
                    domain = domain,
                    facebook_interactions = facebook_interactions,
                    source = source,
                    label = label
                """
            )

    def labelled_query(self, columns):
        query = (
            f"SELECT {','.join(columns)} "
            f"FROM {self.DAILY_NEWS_TABLE} "
        )
        with duckdb.connect(self.db_path) as con:
            results_filtered = con.sql(query).to_df()
        return results_filtered

    def update_labels(self, df):
        with duckdb.connect(self.db_path) as con:
            con.sql(
                f"""
                UPDATE {self.DAILY_NEWS_TABLE} 
                SET label = df_filtered.label 
                FROM (SELECT * FROM df WHERE label IS NOT NULL) as df_filtered
                WHERE {self.DAILY_NEWS_TABLE}.link = df_filtered.link;
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

            label_list = [
                label[0]
                for label in con.sql(
                    f"SELECT DISTINCT label FROM {self.DAILY_NEWS_TABLE} "
                    "WHERE label IS NOT NULL"
                ).fetchall()
            ]
        return {
            "max_fb_interactions": max_fb_interactions,
            "domain_list": domain_list,
            "min_date": min_date,
            "max_date": max_date,
            "labels": label_list,
        }

    def query(self, query):
        with duckdb.connect(self.db_path) as con:
            results_df = con.sql(query).to_df()
        return results_df

    def filtered_query(
        self, columns, domain_filter, min_engagement, date_range, label_filter
    ):
        query = (
            f"SELECT {','.join(columns)} "
            f"FROM {self.DAILY_NEWS_TABLE} "
            f"WHERE domain NOT IN {tuple(domain_filter) if domain_filter else ('NULL',)} "
            f"AND facebook_interactions >= {min_engagement} "
            f"AND published BETWEEN '{date_range[0]}' AND '{date_range[1]}' "
            + (
                ""
                if label_filter is None
                else "AND label is NOT NULL "
                if label_filter == "All, excluding unlabelled"
                else f"AND label='{label_filter}' "
            )
            + f"ORDER BY facebook_interactions DESC "
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
                "modified": datetime.fromtimestamp(file.stat().st_mtime),
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
        """Convert folder to zip, return zip file"""
        raise NotImplementedError

    def __repr__(self):
        return f"FileHandler({repr(self.data_dir)})"

    def __str__(self):
        return f"{[file.name for file in self.raw_data_dir.iterdir()]}"

    def __len__(self):
        return len(list(self.raw_data_dir.iterdir()))




class WeeklyFileHandler:
    WEEKLY_NEWS_TABLE = "weekly_news"

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.weekly_data_dir = self.data_dir / "weekly"
        self.db_path = str(self.data_dir / "weekly_news.db")

        self.weekly_data_dir.mkdir(parents=True, exist_ok=True)

        with duckdb.connect(self.db_path) as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                {self.WEEKLY_NEWS_TABLE}(
                    "published" TIMESTAMP(6),
                    "link" VARCHAR,  
                    "facebook_interactions" BIGINT,
                    "date_extracted" TIMESTAMP(6),
                    PRIMARY KEY ("published", "link", "date_extracted"),
                    "timestamp" TIMESTAMP(6),
                    "source" VARCHAR,
                );
                """
            )

    @staticmethod
    def preprocess_weekly_scan(file) -> pd.DataFrame:
        data_name = file.name
        weekly_data = pd.read_excel(file)
        columns = ["Published", "Link URL", "Facebook Interactions"]
        weekly_data = weekly_data[columns].rename(columns={"Published": "published", "Link URL": "link", "Facebook Interactions": "facebook_interactions"})
        weekly_data['published'] = pd.to_datetime(weekly_data['published'])
        date_string = data_name.partition("posts-")[2].partition(" -")[0]
        format = '%m_%d_%y-%H_%M'
        formatted_date = datetime.strptime(date_string, format)
        weekly_data['date_extracted'] = formatted_date
        weekly_data['timestamp'] = formatted_date
        weekly_data = weekly_data.dropna()
        weekly_data['source'] = data_name

        return weekly_data

    def write_db(self, file) -> None:
        processed_table = self.preprocess_weekly_scan(file)

        with duckdb.connect(self.db_path) as con:
            #
            con.sql(f"DELETE FROM {self.WEEKLY_NEWS_TABLE} WHERE source = '{file.name}'")
            # Append to table, replace if existing link found
            con.sql(
                f"""
                INSERT INTO {self.WEEKLY_NEWS_TABLE} 
                SELECT published, link, facebook_interactions, date_extracted, timestamp, source
                FROM processed_table
                ON CONFLICT (published, link, date_extracted)
                DO UPDATE
                    SET facebook_interactions = facebook_interactions,
                    timestamp = timestamp,
                    source = source
                """
            )

    def query(self):
        query = (
            f"SELECT *"
            f"FROM {self.WEEKLY_NEWS_TABLE} "
        )
        with duckdb.connect(self.db_path) as con:
            results_filtered = con.sql(query).to_df()
        return results_filtered

    def list_csv_filenames(self):
        return [file.name for file in self.weekly_data_dir.iterdir()]

    def list_csv_files_df(self):
        raw_files_info = [
            {
                "filename": file.name,
                "modified": datetime.fromtimestamp(file.stat().st_mtime),
                # "filesize": f"{file.stat().st_size / 1000 / 1000:.2f} MB",
            }
            for file in self.weekly_data_dir.iterdir()
        ]
        return pd.DataFrame(raw_files_info)

    def write_csv(self, file):
        filepath = self.weekly_data_dir / file.name
        filepath.write_bytes(file.getbuffer())
        return filepath

    def remove_files(self, filenames):
        for filename in filenames:
            (self.weekly_data_dir / filename).unlink(missing_ok=True)
            with duckdb.connect(self.db_path) as con:
                con.sql(
                    f"DELETE FROM {self.WEEKLY_NEWS_TABLE} WHERE source = '{filename}'"
                )
        return [self.weekly_data_dir / filename for filename in filenames]