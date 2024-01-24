import asyncio
import datetime
import json
import os
import shutil
from pathlib import Path

import duckdb
import gcsfs
import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

GSHEET_TAXONOMY_ID = os.environ.get("GSHEET_TAXONOMY_ID")
gsheet_taxonomy_url = "https://docs.google.com/spreadsheets/d/" + GSHEET_TAXONOMY_ID

GCS_BUCKET = os.environ["GCS_BUCKET"]
service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
fs = gcsfs.GCSFileSystem(
    project=service_account_info["project_id"], token=service_account_info
)

DATA_DIR = Path("data")

MODEL_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def fetch_latest_themes():
    themes_df = (
        pd.read_csv(f"{gsheet_taxonomy_url}/export?format=csv&gid=92379333")["Theme"]
        .dropna()
        .str.casefold()
        .drop_duplicates()
        .sort_values()
    )
    # Cache taxonomy versions
    taxonomy_folder = DATA_DIR / "taxonomy" / datetime.date.today().isoformat()
    taxonomy_folder.mkdir(parents=True, exist_ok=True)
    themes_df.to_csv(
        f"{taxonomy_folder / 'themes'}.csv",
        index=False,
    )
    return themes_df


def fetch_latest_index():
    index_df = (
        pd.read_csv(f"{gsheet_taxonomy_url}/export?format=csv&gid=755202117")["Index"]
        .dropna()
        .str.casefold()
        .drop_duplicates()
        .sort_values()
    )
    # Cache taxonomy versions
    taxonomy_folder = DATA_DIR / "taxonomy" / datetime.date.today().isoformat()
    taxonomy_folder.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(
        f"{taxonomy_folder / 'indexes'}.csv",
        index=False,
    )
    return index_df


def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
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


def load_classification_model(model_path=None):
    from setfit import SetFitModel

    latest_model_path = get_latest_model_path()
    remove_old_models()
    if model_path is None:
        model_path = latest_model_path

    model = SetFitModel.from_pretrained(model_path)
    return model


def label_df(df: pd.DataFrame, model, column: str) -> pd.DataFrame:
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


def embed_df(df: pd.DataFrame, model, column: str) -> pd.DataFrame:
    embedded_df = df.assign(vector=model.encode(df[column]).tolist())
    return embedded_df


async def async_get_thumbnail_links(urls):
    headers = {
        "User-Agent": "",
    }
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [client.get(url) for url in urls]
        results = await asyncio.gather(*tasks)
    image_urls = []
    for response in results:
        try:
            # print(response.status_code)
            soup = BeautifulSoup(response.text, features="html.parser")
            image_url = soup.find("meta", attrs={"property": "og:image"})
            image_urls.append(image_url["content"])
        except Exception as e:
            image_urls.append(None)
    return image_urls


class FileHandler:
    # Table to display latest metrics & data for users to view
    NEWS_DATA = "news_data"
    # Table to track user labelling for news articles
    NEWS_LABELS = "news_labels"

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw_upload"
        self.db_path = str(self.data_dir / "news.db")

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        with duckdb.connect(self.db_path) as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                {self.NEWS_DATA} (
                    "published" TIMESTAMP,
                    "headline" VARCHAR,
                    "summary" VARCHAR,
                    "link" VARCHAR,
                    "facebook_link" VARCHAR PRIMARY KEY,
                    "facebook_page_name" VARCHAR,
                    "domain" VARCHAR,
                    "facebook_interactions" BIGINT,
                    "source" VARCHAR
                );
                """
            )
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                {self.NEWS_LABELS} (
                    "link" VARCHAR,
                    "facebook_link" VARCHAR PRIMARY KEY,
                    "themes" VARCHAR[],
                    "indexes" VARCHAR[],
                    "subindex" VARCHAR
                );
                """
            )

    @staticmethod
    def preprocess_daily_scan(file) -> pd.DataFrame:
        df = (
            pd.read_csv(
                file,
                usecols=[
                    "Published",
                    "Headline",
                    "Summary",
                    "Link",
                    "Link URL",
                    "Facebook Page Name",
                    "Domain",
                    "Facebook Interactions",
                ],
                dtype={
                    "Published": "string",
                    "Headline": "string",
                    "Summary": "string",
                    "Link": "string",
                    "Link URL": "string",
                    "Facebook Page Name": "string",
                    "Domain": "string",
                    "Facebook Interactions": "int",
                },
            )
            .assign(
                Domain=lambda s: s["Domain"].fillna("facebook.com"),
                Published=lambda s: pd.to_datetime(
                    s["Published"], utc=True
                ).dt.tz_convert("Asia/Singapore"),
            )
            # Sanitize column names to prevent database parse issues
            .rename(lambda col_name: col_name.lower().replace(" ", "_"), axis="columns")
            .rename(columns={"link": "facebook_link", "link_url": "link"})
            .dropna(subset=["link"])
        )

        return df

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

    def write_daily_scan(self, file) -> None:
        processed_table = self.preprocess_daily_scan(file).assign(source=file.name)

        with duckdb.connect(self.db_path) as con:
            con.register("processed_table", processed_table)
            # If duplicate filename, replace all rows
            con.sql(f"DELETE FROM {self.NEWS_DATA} WHERE source = '{file.name}'")
            # If different filename, but different data, update only relevant rows
            con.sql(
                f"""
                INSERT INTO {self.NEWS_DATA}
                SELECT published, 
                    headline, 
                    summary, 
                    link, 
                    facebook_link, 
                    facebook_page_name, 
                    domain, 
                    facebook_interactions, 
                    source 
                FROM processed_table
                ON CONFLICT
                DO UPDATE
                    SET published = EXCLUDED.published,
                        headline = EXCLUDED.headline,
                        summary = EXCLUDED.summary,
                        facebook_page_name = EXCLUDED.facebook_page_name,
                        domain = EXCLUDED.domain,
                        facebook_interactions = EXCLUDED.facebook_interactions,
                        link = EXCLUDED.link,
                        source = EXCLUDED.source,
                            WHERE EXCLUDED.facebook_interactions >= facebook_interactions
                """
            )

    def write_labelled_articles(self, file) -> None:
        processed_table = (
            pd.read_csv(
                file,
                usecols=["link", "facebook_link", "themes", "indexes", "subindex"],
            )
            .dropna(subset=["subindex", "indexes", "indexes"], how="all")
            .assign(
                themes=lambda r: r["themes"].str.split(","),
                indexes=lambda r: r["indexes"].str.split(","),
            )
        )
        self.update_labels(processed_table)

    def query(self, query):
        with duckdb.connect(self.db_path) as con:
            results_df = con.sql(query).to_df()
        return results_df

    def full_query(self):
        query = f"SELECT *" f"FROM {self.NEWS_DATA} "
        return self.query(query)

    def update_subindex(self, df):
        with duckdb.connect(self.db_path) as con:
            con.register("df", df)
            con.sql(
                f"""
                UPDATE {self.NEWS_LABELS} 
                SET subindex = df.subindex,
                FROM df
                WHERE {self.NEWS_LABELS}.facebook_link = df.facebook_link;
                """
            )

    def update_labels(self, df):
        with duckdb.connect(self.db_path) as con:
            con.register("df", df)
            con.sql(
                f"""
                DELETE FROM {self.NEWS_LABELS}
                WHERE {self.NEWS_LABELS}.facebook_link IN (SELECT df.facebook_link FROM df)
                """
            )

            con.sql(
                f"""
                INSERT INTO {self.NEWS_LABELS}
                SELECT link, facebook_link, themes, indexes, subindex
                FROM df
                """
            )

    def get_filter_bounds(self):
        with duckdb.connect(self.db_path) as con:
            max_fb_interactions = con.sql(
                f"SELECT MAX(facebook_interactions) FROM {self.NEWS_DATA}"
            ).fetchone()[0]

            domain_list = [
                domain
                for domain, count in con.sql(
                    "SELECT domain, COUNT(*) AS count FROM "
                    f"{self.NEWS_DATA} "
                    "GROUP BY domain "
                    "ORDER BY count DESC"
                ).fetchall()
            ]

            min_date, max_date = con.sql(
                f"SELECT MIN(published), MAX(published) FROM " f"{self.NEWS_DATA} "
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

    def filtered_query(
        self,
        domain_filter,
        min_engagement,
        date_range,
    ):
        query = f"""
            SELECT * FROM {self.NEWS_DATA}
            LEFT JOIN {self.NEWS_LABELS} 
                ON {self.NEWS_DATA}.facebook_link = {self.NEWS_LABELS}.facebook_link
            WHERE domain NOT IN {tuple(domain_filter) if domain_filter else (' ',)}
            AND facebook_interactions >= {min_engagement} 
            AND published BETWEEN '{date_range[0]}' AND '{date_range[1]}' 
            AND headline is not NULL 
            ORDER BY facebook_interactions DESC
            """
        with duckdb.connect(self.db_path) as con:
            results_filtered = con.sql(query).to_df()
        return results_filtered

    def labelled_query(
        self,
    ):
        query = f"""
            SELECT * FROM {self.NEWS_DATA}
            LEFT JOIN {self.NEWS_LABELS} 
                ON {self.NEWS_DATA}.facebook_link = {self.NEWS_LABELS}.facebook_link
            WHERE ((themes IS NOT NULL) AND (len(themes) > 0))
            OR ((indexes IS NOT NULL) AND (len(indexes) > 0))
            OR subindex IS NOT NULL
            ORDER BY published ASC
            """
        with duckdb.connect(self.db_path) as con:
            results_labelled = con.sql(query).to_df()
        return results_labelled

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
                con.sql(f"DELETE FROM {self.NEWS_DATA} WHERE source = '{filename}'")
        return [self.raw_data_dir / filename for filename in filenames]

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
