from pathlib import Path

from modelling import delete_old_models, train_model, train_test_split_eval_model
from rocketry import Rocketry
from rocketry.conds import after_success, time_of_day, weekly

app = Rocketry()


@app.cond()
def file_exists(file):
    return Path(file).exists()


@app.task(weekly.on("Sat") & time_of_day.after("6:00"))
def train_model_weekly():
    print("Start weekly training")
    train_model()


@app.task(after_success(train_model_weekly))
def eval_model_after_training():
    train_test_split_eval_model()


@app.task(after_success(eval_model_after_training))
def delete_old_model_after_eval():
    delete_old_models()


if __name__ == "__main__":
    app.run()
