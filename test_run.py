from sklearn.model_selection import StratifiedKFold
from mlmodels import LogisticRegression, RandomForest, LinearRegression
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import SMOTE, RandomOverSampler

import joblib
import numpy as np
import pandas as pd
import wandb
import yaml



CONFIG = [
    {"Model": RandomForest, "Antibiotic": "Erythromycin", "Scaler": None, "Resampler": None, "config": {"class_weight": "balanced", "n_estimators": 500}},
    {"Model": RandomForest, "Antibiotic": "Ciprofloxacin", "Scaler": None, "Resampler": None, "config": {"class_weight": None, "n_estimators": 250}},
    {"Model": RandomForest, "Antibiotic": "Erythromycin", "Scaler": Normalizer("l2"), "Resampler": None, "config": {"class_weight": "balanced", "n_estimators": 250}},
    {"Model": RandomForest, "Antibiotic": "Ciprofloxacin", "Scaler": Normalizer("l2"), "Resampler": None, "config": {"class_weight": None, "n_estimators": 250} },
    {"Model": RandomForest, "Antibiotic": "Erythromycin", "Scaler": Normalizer("l2"), "Resampler": "GAN", "config": {"class_weight": "balanced", "n_estimators": 500}},
    {"Model": RandomForest, "Antibiotic": "Ciprofloxacin", "Scaler": Normalizer("l2"), "Resampler": "GAN", "config": {"class_weight": "balanced", "n_estimators": 250}},
    {"Model": RandomForest, "Antibiotic": "Erythromycin", "Scaler": Normalizer("l2"), "Resampler": "SMOTE", "config": {"class_weight": "balanced", "n_estimators": 500}},
    {"Model": RandomForest, "Antibiotic": "Ciprofloxacin", "Scaler": Normalizer("l2"), "Resampler": "SMOTE", "config": {"class_weight": None, "n_estimators": 500}},
    {"Model": RandomForest, "Antibiotic": "Erythromycin", "Scaler": Normalizer("l2"), "Resampler": "Random", "config": {"class_weight": "balanced", "n_estimators": 250}},
    {"Model": RandomForest, "Antibiotic": "Ciprofloxacin", "Scaler": Normalizer("l2"), "Resampler": "Random", "config": {"class_weight": None, "n_estimators": 250}},
]



def setup_dataset(antibiotic):
    df = pd.read_csv(f"./data/test_{antibiotic[0]}.csv")
    X_test = df.drop(antibiotic, axis=1).values
    y_test = df[antibiotic].values

    df = pd.read_csv(f"./data/train_{antibiotic[0]}.csv")
    X_train = df.drop(antibiotic, axis=1).values
    y_train = df[antibiotic].values
    return X_train, X_test, y_train, y_test

def train(config):

    model_class = config["Model"]
    scaling = config["Scaler"]
    resampler = config["Resampler"]
    antibiotic = config["Antibiotic"]
    model_config = config["config"]

    print(scaling)

    if not scaling:
        scaling_name = None
    else:
        scaling_name = "L2"

    wandb.init(name = f"{model_class.__name__}_{antibiotic[0]}_{scaling_name}_{resampler}", config =  
                     {"Model": model_class.__name__, "Scaler": scaling_name, "Resampler": resampler, "Antibiotic": antibiotic})
    metrics = {"AUC_ROC": [], "AUC_PR": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    X_train, X_test, y_train, y_test = setup_dataset(antibiotic)
    
    if scaling:
        X_train = scaling.fit_transform(X_train)
        X_test = scaling.transform(X_test)

    print(X_test.shape, y_test.shape)

    X_train, y_train = resample(X_train, y_train, resampler, antibiotic)
    model = model_class(model_config)
    model.fit(X_train, y_train, None, None)
    metrics = model.evaluate(X_test, y_test, metrics)
    for metric, metric_list in metrics.items():
        wandb.log({metric: np.mean(metric_list)})

    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)

    wandb.log({"ROC": wandb.sklearn.plot_roc(y_test, y_probas, classes_to_plot=[1.]),
                   "Precision-Recall": wandb.sklearn.plot_precision_recall(y_test, y_probas, classes_to_plot=[1.]),
                   "Confusion": wandb.sklearn.plot_confusion_matrix(y_test, y_pred)})

    wandb.finish()

def resample(x, y, resampler, antibiotic):
    num_1 = np.sum(x == 1)
    num_0 = np.sum(y == 0)
    missing_samples = num_0 - num_1
    if resampler == 'GAN':
        fake = joblib.load(f"data/fake_{antibiotic[0]}.pkl")
        indices = np.random.permutation(fake.shape[0])[:missing_samples]
        selected_fake = fake[indices]
        labels = np.ones(selected_fake.shape[0])
        x = np.concatenate([x, selected_fake], axis=0)
        y = np.concatenate([y, labels], axis=0)
    elif resampler == 'SMOTE':
        smote = SMOTE(random_state = 42)
        x, y = smote.fit_resample(x, y)
    elif resampler == 'RANDOM':
        ros = RandomOverSampler(random_state=42)
        x, y = ros.fit_resample(x, y)
    return x, y

if __name__ == "__main__":
    for config in CONFIG:
        train(config)