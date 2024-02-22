from sklearn.model_selection import StratifiedKFold
from mlmodels import LogisticRegression, RandomForest, MLP_TF, CNN_TF

import numpy as np
import tensorflow as tf
import pandas as pd
import wandb
import yaml

CONFIG_FILE = "./sweep_configs/MLP_Erythromycin.yaml"
MODEL = MLP_TF
ANTIBIOTIC = "Erythromycin"
N_SPLITS = 5


def setup_dataset():
    df = pd.read_csv(f"./data/train_{ANTIBIOTIC[0]}.csv")
    X = df.drop(ANTIBIOTIC, axis=1).values
    y = df[ANTIBIOTIC].values
    return X, y

X, y = setup_dataset()

def train():
    run = wandb.init()
    config = run.config.as_dict()
    metrics = {"AUC_ROC": [], "AUC_PR": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    skf = StratifiedKFold(n_splits = N_SPLITS, shuffle = True, random_state = 42)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
        model = MODEL(config)
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_val, y_val, metrics)
    for metric, metric_list in metrics.items():
        wandb.log({metric: np.mean(metric_list)})


if __name__ == "__main__":
    with open(CONFIG_FILE, 'r') as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train)