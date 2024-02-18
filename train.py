import wandb
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import joblib
import os
from models import MLP, CNN
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from train_models import fit, test, plot_metrics
from dataset_mng import create_and_split_dataset
from sklearn.metrics import classification_report, confusion_matrix


def train_model(X, Y, model, model_params, training_params, eval_type, type):
    torch.manual_seed(0)
    for key, value in model_params.items():
        if value == 'None':
            model_params[key] = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
    pytorch_models = ['mlp', 'cnn']
    if isinstance(X, tuple):
        Y_train, Y_val, Y_test = Y
        X_train, X_val, X_test = X
    else:
        X_train, X_val, X_test = X, X, X
        Y_train, Y_val, Y_test = Y, Y, Y
    Y_val = torch.from_numpy(Y_val.values).to(device, dtype = torch.float32)
    X_val = torch.from_numpy(X_val.values).to(device, dtype = torch.float32)
    Y_train = torch.from_numpy(Y_train.values).to(device, dtype = torch.float32)
    X_train = torch.from_numpy(X_train.values).to(device, dtype = torch.float32)
    Y_test = torch.from_numpy(Y_test.values).to(device, dtype = torch.float32)
    X_test = torch.from_numpy(X_test.values).to(device, dtype = torch.float32)

    if model == 'random_forest':
        model_params['n_jobs'] = 4
        model_params['class_weight'] = 'balanced'
        clf = RandomForestClassifier(**model_params, random_state=0)

    elif model == 'logistic_regression':
        model_params['class_weight'] = 'balanced'
        model_params['n_jobs'] = 4
        model_params['penalty'] = 'l2'
        clf = LogisticRegression(**model_params, random_state=0)

    if model not in pytorch_models:
        if eval_type == 'kfolds':
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            accuracies = []
            precisions = []
            aucs = []

            for train_index, test_index in kf.split(X, Y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

                clf.fit(X_train, Y_train)
                Y_pred = clf.predict(X_test)

                accuracy = accuracy_score(Y_test, Y_pred)
                precision = precision_score(Y_test, Y_pred)

                auc = roc_auc_score(Y_test, clf.predict_proba(X_test)[:, 1])

                accuracies.append(accuracy)
                precisions.append(precision)
                aucs.append(auc)

            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_auc = sum(aucs) / len(aucs)

            return avg_accuracy, avg_precision, avg_auc
        
        elif eval_type == 'split_test':
            X_train, X_val, X_test = X
            Y_train, Y_val, Y_test = Y

            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_val)

            print(classification_report(Y_val, Y_pred))
            print(confusion_matrix(Y_val, Y_pred))
            accuracy = accuracy_score(Y_val, Y_pred)
            precision = precision_score(Y_val, Y_pred)
            auc = roc_auc_score(Y_val, clf.predict_proba(X_val)[:, 1])

            return accuracy, precision, auc
        
    else:
        try:
            model_params['input_size'] = X.values.shape
        except: 
            model_params['input_size'] = X[0].values.shape
        model_params['output_size'] = 1


        num_positives = torch.sum(Y_train, dim=0)
        num_negatives = len(Y_train) - num_positives
        pos_weight  = num_negatives / num_positives

        epochs = 200
        patience = 5
        batch_size = 32

        if eval_type == 'split_test':
            if model == 'mlp':
                clf = MLP(**model_params).to(device)
            elif model == 'cnn':
                clf = CNN(**model_params).to(device)
            
            if training_params['weight']=='None':
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.BCEWithLogitsLoss(pos_weight)
            optimizer = optim.Adam(clf.parameters(), lr=0.0005, weight_decay = training_params['l2'])
            train_set, val_set, test_set = create_and_split_dataset(X, Y)

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 16)
            val_loader = DataLoader(val_set, batch_size=batch_size, num_workers = 16)
            test_loader = DataLoader(test_set, batch_size=batch_size, num_workers = 16)

            train_losses, val_losses = fit(clf, train_loader, val_loader, criterion, optimizer, epochs, patience, device, f'{model}_{type}')
            plot_metrics((train_losses, val_losses), f'{model}_{type}')
            clf.load_state_dict(torch.load(f"models/{model}_{type}.pth"))
            test(clf, test_loader, criterion, device, f'{model}_{type}')

            clf.eval()
            with torch.no_grad():
                y_pred = torch.sigmoid(clf(X_val)).to(dtype = torch.float32)

            Y_val, y_pred = Y_val.cpu(), y_pred.cpu()
            auc = roc_auc_score(Y_val, y_pred)

            for i in range(y_pred.shape[0]):
                if y_pred[i]>=0.5:
                    y_pred[i] = 1.0
                else:
                    y_pred[i] = 0.0
            
            print(classification_report(Y_val, y_pred))
            print(confusion_matrix(Y_val, y_pred))
            accuracy = accuracy_score(Y_val, y_pred)
            precision = precision_score(Y_val, y_pred)

            return accuracy, precision, auc
        
        else:
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            accuracies = []
            precisions = []
            aucs = []

            X = torch.from_numpy(X.values)
            Y = torch.from_numpy(Y.values)

            for train_index, val_index in kf.split(X, Y):
                if model == 'mlp':
                    clf = MLP(**model_params).to(device)
                elif model == 'cnn':
                    clf = CNN(**model_params).to(device)

                if training_params['weight']=='None':
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.BCEWithLogitsLoss(pos_weight)
                optimizer = optim.Adam(clf.parameters(), lr=0.0005, weight_decay = training_params['l2'])

                X_train_fold, X_val_fold = X[train_index], X[val_index]
                Y_train_fold, Y_val_fold = Y[train_index], Y[val_index]

                train_set_fold = TensorDataset(X_train_fold, Y_train_fold)
                val_set_fold = TensorDataset(X_val_fold, Y_val_fold)

                train_loader_fold = DataLoader(train_set_fold, batch_size=batch_size, shuffle=True, num_workers=16)
                val_loader_fold = DataLoader(val_set_fold, batch_size=batch_size, num_workers=16)

                train_losses, val_losses = fit(clf, train_loader_fold, val_loader_fold, criterion, optimizer, epochs, patience, device, f'{model}')

                clf.eval()
                with torch.no_grad():
                    y_pred_fold = torch.sigmoid(clf(X_val_fold.to(device=device, dtype=torch.float32))).to(dtype=torch.float32)


                auc_fold = roc_auc_score(Y_val_fold, y_pred_fold.cpu())

                for i in range(y_pred_fold.shape[0]):
                    if y_pred_fold[i]>=0.5:
                        y_pred_fold[i] = 1
                    else:
                        y_pred_fold[i] = 0
                accuracy_fold = accuracy_score(Y_val_fold, y_pred_fold.cpu())
                precision_fold = precision_score(Y_val_fold, y_pred_fold.cpu())

                accuracies.append(accuracy_fold)
                precisions.append(precision_fold)
                aucs.append(auc_fold)

            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_auc = sum(aucs) / len(aucs)

            return avg_accuracy, avg_precision, avg_auc





