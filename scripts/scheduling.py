from rocketry import Rocketry

from pathlib import Path
from rocketry.conds import weekly, time_of_day
from modelling import eval_model, train_model, delete_old_models


app = Rocketry()


@app.cond()
def file_exists(file):
    return Path(file).exists()


@app.task(
    weekly.on("Sat")
    & time_of_day.after("6:00")
    & file_exists("data/train/generated_training_data.parquet")
)
def train_model_weekly():
    print("Start daily training")
    train_model()
    eval_model()
    print("Complete daily training")
    delete_old_models()
    print("Delete old models")


if __name__ == "__main__":
    app.run()
