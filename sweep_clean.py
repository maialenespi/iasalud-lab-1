from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer
from mlmodels import LogisticRegression, RandomForest, MLP_TF, CNN_TF
from imblearn.over_sampling import SMOTE, RandomOverSampler

import numpy as np
import tensorflow as tf
import pandas as pd
import wandb
import yaml
import joblib

MODEL = MLP_TF
ANTIBIOTIC = "Erythromycin"
ALGORITHM = 'MLP'
CONFIG_FILE = f"./sweep_configs/{ALGORITHM}_{ANTIBIOTIC}.yaml"
N_SPLITS = 5
RESAMPLER = 'GAN'
SCALING = Normalizer(norm='l2')

def setup_dataset():
    df = pd.read_csv(f"./data/train_{ANTIBIOTIC[0]}.csv")
    X = df.drop(ANTIBIOTIC, axis=1).values
    y = df[ANTIBIOTIC].values
    return X, y

X, y = setup_dataset()
if SCALING != 'None':
    X = SCALING.fit_transform(X)

def train():
    run = wandb.init()
    config = run.config.as_dict()
    metrics = {"AUC_ROC": [], "AUC_PR": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    skf = StratifiedKFold(n_splits = N_SPLITS, shuffle = True, random_state = 42)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
        X_train, y_train = resample(X_train, y_train)
        model = MODEL(config)
        model.fit(X_train, y_train, X_val, y_val)
        metrics = model.evaluate(X_val, y_val, metrics)
    for metric, metric_list in metrics.items():
        wandb.log({metric: np.mean(metric_list)})

def resample(x, y):
    num_1 = np.sum(x == 1)
    num_0 = np.sum(y == 0)
    missing_samples = num_0 - num_1
    if RESAMPLER == 'GAN':
        fake = joblib.load(f"fake_{ANTIBIOTIC[0]}.pkl")
        indices = np.random.permutation(fake.shape[0])[:missing_samples]
        selected_fake = fake[indices]
        labels = np.ones(selected_fake.shape[0])
        x = np.concatenate([x, selected_fake], axis=0)
        y = np.concatenate([y, labels], axis=0)
    elif RESAMPLER == 'SMOTE':
        smote = SMOTE(random_state = 42)
        x, y = smote.fit_resample(x, y)
    elif RESAMPLER == 'RANDOM':
        ros = RandomOverSampler(random_state=42)
        x, y = ros.fit_resample(x, y)
    return x, y


if __name__ == "__main__":
    with open(CONFIG_FILE, 'r') as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train)