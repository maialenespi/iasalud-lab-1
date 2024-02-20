from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import models
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
import pandas as pd
import wandb
import yaml

CONFIG_FILE = "./sweep_configs/MLP_Erythromycin.yaml"
MODEL, ANTIBIOTIC = CONFIG_FILE.split("/")[2].split(".")[0].split("_")

def setup_dataset():
    df = pd.read_csv(f"train_{ANTIBIOTIC[0]}.csv")
    X = df.drop(ANTIBIOTIC, axis=1).values
    y = df[ANTIBIOTIC].values
    return X, y

X, y = setup_dataset()
    
def evaluate(y_val, y_pred, y_proba, metrics):
    metrics["AUC_ROC"].append(roc_auc_score(y_val, y_proba))
    metrics["AUC_PR"].append(average_precision_score(y_val, y_proba))
    metrics["Accuracy"].append(accuracy_score(y_val, y_pred))
    metrics["Precision"].append(precision_score(y_val, y_pred))
    metrics["Recall"].append(recall_score(y_val, y_pred))
    metrics["F1"].append(f1_score(y_val, y_pred))
    return metrics

def train():
    run = wandb.init()
    config = run.config
    metrics = {"AUC_ROC": [], "AUC_PR": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
        
        if MODEL in ["LogisticRegression", "RandomForest"]:
            if MODEL == "LogisticRegression":
                model = LogisticRegression(random_state = 42, penalty = config.penalty, class_weight = config.class_weight)
            elif MODEL == "RandomForest":
                model = RandomForestClassifier(random_state = 42, n_estimators = config.n_estimators, class_weight = config.class_weight)
            model.fit(X_train, y_train)
            y_pred, y_proba = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

        elif MODEL in ["MLP", "CNN"]:
            if MODEL == "MLP":

                # TODO: Leaky RELU
                model = models.Sequential()
                model.add(Dense(config.hidden_dim, activation=config.hidden_act, input_dim=X_train.shape[1]))
                for _ in range(config.n_layers):
                    model.add(Dense(config.hidden_dim, activation=config.hidden_act))
                model.add(Dense(1, activation='sigmoid'))

            elif MODEL == "CNN":
                model = models.Sequential()
                model.add(Conv1D(config.hidden_dim, kernel_size=config.kernel_size, activation=config.hidden_act, input_shape=X_train.shape))
                model.add(Flatten())
                model.add(Dense(1, activation='sigmoid'))
            
            # TODO: EARLY STOPPING + EPOCHS
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            class_weight = {}
            class_weight[0], class_weight[1]  = compute_class_weight(class_weight=config.class_weight, classes=np.unique(y_val), y=y_val)
            model.fit(X_train, y_train, epochs=100, class_weight=class_weight, verbose=0)
            y_proba = model.predict(X_val)
            y_pred = tf.round(y_proba)
        
        metrics = evaluate(y_val, y_pred, y_proba, metrics)
    for metric, metric_list in metrics.items():
        wandb.log({metric: np.mean(metric_list)})


if __name__ == "__main__":
    with open(CONFIG_FILE, 'r') as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train)