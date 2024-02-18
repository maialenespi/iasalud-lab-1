from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import joblib
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_and_split_dataset(X, Y):

    # Crear y dividir datasets de PyTorch
    if isinstance(X, tuple):
        dataset_train = CustomDataset(X[0], Y[0])
        dataset_val = CustomDataset(X[1], Y[1])
        dataset_test = CustomDataset(X[2], Y[2])
    else:
        dataset_train = CustomDataset(X, Y)
        dataset_val = CustomDataset(X, Y)
        dataset_test = CustomDataset(X, Y)

    return dataset_train, dataset_val, dataset_test

def create_dataset():
    if os.path.exists('dataset.pkl'):
        dataset = joblib.load('dataset.pkl')
    else:
        df = pd.read_csv("practica_micro.csv")
        df['MALDI_binned'] = df['MALDI_binned'].apply(json.loads)
        df = pd.concat([pd.DataFrame(df["MALDI_binned"].to_dict()).T, df], axis=1)
        df.drop("MALDI_binned", inplace=True, axis=1)
        df.columns = [f"MALDI_{i}" for i in range(6000)] + ["Erythromycin", "Ciprofloxacin"]
        dataset = df
        joblib.dump(df, 'dataset.pkl')

    return dataset

def split_dataset(dataset):
    train, validate, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))])

    y_train_E = train["Erythromycin"]
    y_val_E = validate["Erythromycin"]
    y_test_E = test["Erythromycin"]

    y_train_C = train["Ciprofloxacin"]
    y_val_C = validate["Ciprofloxacin"]
    y_test_C = test["Ciprofloxacin"]

    X_train = train.drop(["Erythromycin","Ciprofloxacin"], axis=1)
    X_val = validate.drop(["Erythromycin","Ciprofloxacin"], axis=1)
    X_test = test.drop(["Erythromycin","Ciprofloxacin"], axis=1)

    return (X_train, X_val, X_test), ((y_train_E, y_val_E, y_test_E), (y_train_C, y_val_C, y_test_C))