import datetime
import json
import os
import shutil
from pathlib import Path

import duckdb
import gcsfs
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel

load_dotenv()

GSHEET_TAXONOMY_ID = os.environ.get("GSHEET_TAXONOMY_ID")
gsheet_taxonomy_url = "https://docs.google.com/spreadsheets/d/" + GSHEET_TAXONOMY_ID

GCS_BUCKET = os.environ["GCS_BUCKET"]
service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
fs = gcsfs.GCSFileSystem(
    project=service_account_info["project_id"], token=service_account_info
)

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
WEEKLY_DATA_DIR = DATA_DIR / "weekly"


def fetch_latest_taxonomy() -> pd.DataFrame:
    taxonomy_df = (
        pd.read_csv(gsheet_taxonomy_url + "/export?format=csv")[["Theme", "Index"]][:-1]
        .dropna(axis=0, how="all")
        .assign(
            Theme=lambda row: row["Theme"].str.casefold(),
            Index=lambda row: row["Index"].str.casefold(),
        )
        .ffill()
        .drop_duplicates()
        .sort_values(["Theme", "Index"])
        .reset_index(drop=True)
    )

    # Cache taxonomy versions
    taxonomy_folder = DATA_DIR / "taxonomy"
    taxonomy_folder.mkdir(parents=True, exist_ok=True)
    taxonomy_df.to_csv(
        f"{taxonomy_folder / datetime.date.today().isoformat()}.csv",
        index=False,
    )
    return taxonomy_df


def load_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer(
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        cache_folder="cached_models",
    )
    return model


def get_latest_model_path():
    latest_model_uri = "gs://" + sorted(fs.ls(f"gs://{GCS_BUCKET}/trained_models"))[-1]
    latest_model_path = latest_model_uri.lstrip(f"gs://{GCS_BUCKET}/")

    if latest_model_path not in [
        str(model_path) for model_path in Path("trained_models").glob("20*")
    ]:
        fs.get(latest_model_uri, latest_model_path, recursive=True)

    return latest_model_path


def remove_old_models(models_to_keep=12):
    for old_model_path in sorted(Path("trained_models").glob("20*"))[:-models_to_keep]:
        shutil.rmtree(old_model_path)


def load_classification_model(model_path=None) -> SetFitModel:
    latest_model_path = get_latest_model_path()
    remove_old_models()
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
                    "facebook_page_name" VARCHAR,
                    "domain" VARCHAR,
                    "facebook_interactions" BIGINT,  
                    "source" VARCHAR,
                    "themes" VARCHAR[],
                    "indexes" VARCHAR[],
                    "subindex" VARCHAR,
                    PRIMARY KEY ("published", "link", "facebook_page_name")
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
                    "Link URL",
                    "Facebook Page Name",
                    "Domain",
                    "Facebook Interactions",
                ],
                dtype={
                    "Published": "string",
                    "Headline": "string",
                    "Summary": "string",
                    "Link URL": "string",
                    "Facebook Page Name": "string",
                    "Domain": "string",
                    "Facebook Interactions": "int",
                },
                # parse_dates=["Published"],
            )
            .assign(source=source, Domain=lambda s: s["Domain"].fillna("facebook.com"))
            .rename(lambda col_name: col_name.lower().replace(" ", "_"), axis="columns")
            .rename(columns={"link_url": "link"})
        )

        df["published"] = pd.to_datetime(df["published"])
        df["published"] = df["published"] + datetime.timedelta(hours=7)
        df = df.dropna(subset=["link"])
        data_name = file.name
        date_string = data_name.partition("posts-")[2].partition(".csv")[0]
        format = "%m_%d_%y-%H_%M"
        latest_date_file = datetime.datetime.strptime(date_string, format).date()

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
                    "facebook_page_name",
                    "domain",
                    "facebook_interactions",
                    "subindex",
                    "source",
                ],
                dtype={
                    "published": "string",
                    "headline": "string",
                    "summary": "string",
                    "link": "string",
                    "facebook_page_name": "string",
                    "domain": "string",
                    "facebook_interactions": "int",
                    "subindex": "string",
                    "source": "string",
                },
            )
        ).copy()
        return df

    def write_csv(self, file):
        filepath = self.raw_data_dir / file.name
        filepath.write_bytes(file.getbuffer())
        return filepath

    def write_db(self, file) -> None:
        processed_table, latest_date_file = self.preprocess_daily_scan(
            file, source=file.name
        )

        query = f"SELECT max(published) max_published " f"FROM {self.DAILY_NEWS_TABLE} "
        with duckdb.connect(self.db_path) as con:
            results_filtered = con.sql(query).to_df()

        latest_date_db = results_filtered["max_published"].dt.date[0]
        if pd.isnull(latest_date_db):
            insert_table = processed_table
        else:
            insert_table = processed_table.loc[
                processed_table["published"].dt.date.between(
                    latest_date_db, latest_date_file, inclusive=False
                )
            ]
        #
        with duckdb.connect(self.db_path) as con:
            #
            con.sql(f"DELETE FROM {self.DAILY_NEWS_TABLE} WHERE source = '{file.name}'")
            # Append to table, replace if existing link found
            con.sql(
                f"""
                INSERT INTO {self.DAILY_NEWS_TABLE}
                SELECT published, headline, summary, link, facebook_page_name, domain, facebook_interactions, source, list_value(), list_value(), NULL
                FROM insert_table
                ON CONFLICT (published, link, facebook_page_name)
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
        min_date = min(processed_table["published"])
        max_date = max(processed_table["published"])
        with duckdb.connect(self.db_path) as con:
            con.sql(
                f"DELETE FROM {self.DAILY_NEWS_TABLE} WHERE published >= '{min_date}' AND published <= '{max_date}'"
            )
            # Append to table, replace if existing link found
            con.sql(
                f"""
                INSERT INTO {self.DAILY_NEWS_TABLE} 
                SELECT published, headline, summary, link, facebook_page_name, domain, facebook_interactions, source, themes, indexes, subindex 
                FROM processed_table
                ON CONFLICT (published, link, facebook_page_name)
                DO UPDATE
                    SET headline = headline,
                    summary = summary,
                    domain = domain,
                    facebook_interactions = facebook_interactions,
                    source = source,
                    themes = themes,
                    indexes = indexes,
                    subindex = subindex
                """
            )

    def labelled_query(self, columns):
        query = f"SELECT {','.join(columns)} " f"FROM {self.DAILY_NEWS_TABLE} "
        with duckdb.connect(self.db_path) as con:
            results_filtered = con.sql(query).to_df()
        return results_filtered

    def full_query(self):
        query = f"SELECT *" f"FROM {self.DAILY_NEWS_TABLE} "
        with duckdb.connect(self.db_path) as con:
            full_df = con.sql(query).to_df()
        return full_df

    def update_subindex(self, df):
        with duckdb.connect(self.db_path) as con:
            con.sql(
                f"""
                UPDATE {self.DAILY_NEWS_TABLE} 
                SET subindex = df_filtered.subindex,
                FROM (SELECT * FROM df WHERE subindex NOT NULL) as df_filtered
                WHERE {self.DAILY_NEWS_TABLE}.link = df_filtered.link;
                """
            )

    def update_themes(self, df):
        with duckdb.connect(self.db_path) as con:
            # con.register("df", df)
            con.sql(
                f"""
                DELETE FROM {self.DAILY_NEWS_TABLE}
                WHERE {self.DAILY_NEWS_TABLE}.link IN (SELECT df.link FROM df)
                """
            )

            con.sql(
                f"""
                INSERT INTO {self.DAILY_NEWS_TABLE}
                SELECT published, headline, summary, link, facebook_page_name, domain, facebook_interactions, source, themes, indexes, subindex
                FROM df
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

            # theme_list = [
            #     theme[0]
            #     for theme in con.sql(
            #         f"SELECT DISTINCT theme FROM {self.DAILY_NEWS_TABLE} "
            #         "WHERE theme IS NOT NULL"
            #     ).fetchall()
            # ]
        return {
            "max_fb_interactions": max_fb_interactions,
            "domain_list": domain_list,
            "min_date": min_date,
            "max_date": max_date,
            # "themes": theme_list,
        }

    def query(self, query):
        with duckdb.connect(self.db_path) as con:
            results_df = con.sql(query).to_df()
        return results_df

    def filtered_query(
        self,
        columns,
        domain_filter,
        min_engagement,
        date_range,
    ):
        query = (
            f"SELECT {','.join(columns)} "
            f"FROM {self.DAILY_NEWS_TABLE} "
            f"WHERE domain NOT IN {tuple(domain_filter) if domain_filter else (' ',)} "
            f"AND facebook_interactions >= {min_engagement} "
            f"AND published BETWEEN '{date_range[0]}' AND '{date_range[1]}' "
            "AND headline is not NULL "
            # + (
            #     ""
            #     if label_filter is None
            #     else "AND theme is NOT NULL "
            #     if label_filter == "All, excluding unthemed"
            #     else f"AND theme='{label_filter}' "
            # )
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
                    "facebook_page_name" VARCHAR,
                    "facebook_interactions" BIGINT,
                    "date_time_extracted" TIMESTAMP(6),
                    PRIMARY KEY ("published", "link", "facebook_page_name", "date_time_extracted"),
                    "source" VARCHAR,
                );
                """
            )

    @staticmethod
    def preprocess_weekly_scan(file) -> pd.DataFrame:
        data_name = file.name
        weekly_data = pd.read_csv(file)
        columns = [
            "Published",
            "Link URL",
            "Facebook Page Name",
            "Facebook Interactions",
        ]
        weekly_data = weekly_data[columns].rename(
            columns={
                "Published": "published",
                "Link URL": "link",
                "Facebook Page Name": "facebook_page_name",
                "Facebook Interactions": "facebook_interactions",
            }
        )
        weekly_data["published"] = pd.to_datetime(weekly_data["published"])
        weekly_data["published"] = weekly_data["published"] + datetime.timedelta(
            hours=7
        )
        date_string = data_name.partition("posts-")[2].partition(".csv")[0]
        format = "%m_%d_%y-%H_%M"
        formatted_date = datetime.datetime.strptime(date_string, format)
        weekly_data["date_time_extracted"] = formatted_date
        weekly_data = weekly_data.dropna()
        weekly_data["source"] = data_name

        return weekly_data

    def write_db(self, file) -> None:
        processed_table = self.preprocess_weekly_scan(file)

        with duckdb.connect(self.db_path) as con:
            #
            con.sql(
                f"DELETE FROM {self.WEEKLY_NEWS_TABLE} WHERE source = '{file.name}'"
            )
            # Append to table, replace if existing link found
            con.sql(
                f"""
                INSERT INTO {self.WEEKLY_NEWS_TABLE}
                SELECT published, link, facebook_page_name, facebook_interactions, date_time_extracted, source
                FROM processed_table
                ON CONFLICT (published, link, facebook_page_name, date_time_extracted)
                DO UPDATE
                    SET facebook_interactions = facebook_interactions,
                    source = source
                """
            )

    # def write_may_june_db(self):
    #     may_june_data = pd.read_excel("data/weekly/may_june_data_filtered.xlsx")
    #
    #     # for i in range(0, may_june_data.shape[0]):
    #     #     row = may_june_data.iloc[[i]]
    #
    #     with duckdb.connect(self.db_path) as con:
    #         #
    #         con.sql(
    #             f"""
    #             INSERT INTO {self.WEEKLY_NEWS_TABLE}
    #             SELECT published, link, facebook_page_name, facebook_interactions, date_time_extracted, source
    #             FROM may_june_data
    #             ON CONFLICT (published, link, facebook_page_name, date_time_extracted)
    #             DO UPDATE
    #                 SET facebook_interactions = facebook_interactions,
    #                 source = source
    #             """
    #         )

    def full_query(self):
        query = f"SELECT *" f"FROM {self.WEEKLY_NEWS_TABLE} "
        with duckdb.connect(self.db_path) as con:
            full_df = con.sql(query).to_df()
        return full_df

    def list_csv_filenames(self):
        return [file.name for file in self.weekly_data_dir.iterdir()]

    def list_csv_files_df(self):
        raw_files_info = [
            {
                "filename": file.name,
                "modified": datetime.datetime.fromtimestamp(file.stat().st_mtime),
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
