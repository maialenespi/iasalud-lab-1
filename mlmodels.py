from sklearn.linear_model import LogisticRegression as LRBase
from sklearn.ensemble import RandomForestClassifier as RFBase
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, f1_score, average_precision_score

import numpy as np
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MLModelSKLearn:
    def __init__(self, **config):
        pass
    
    def fit(self, X_train, y_train, X_val, y_val):
        pass

    def predict(self, X_val):
        pass

    def predict_proba(self, X_val):
        pass

    def evaluate(self, X_val, y_val, metrics):
        y_pred = self.predict(X_val)
        y_proba = self.predict_proba(X_val)
        metrics["AUC_ROC"].append(roc_auc_score(y_val, y_proba))
        metrics["AUC_PR"].append(average_precision_score(y_val, y_proba))
        metrics["Accuracy"].append(accuracy_score(y_val, y_pred))
        metrics["Precision"].append(precision_score(y_val, y_pred))
        metrics["Recall"].append(recall_score(y_val, y_pred))
        metrics["F1"].append(f1_score(y_val, y_pred))
        return metrics

class LogisticRegression(MLModelSKLearn):
    def __init__(self, config):
        super().__init__()
        self.model = LRBase(random_state=42, penalty=config['penalty'], class_weight=config['class_weight'])

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_val):
        return self.model.predict(X_val)
    
    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)[:, 1]
    
class RandomForest(MLModelSKLearn):
    def __init__(self, config):
        super().__init__()
        self.model = RFBase(random_state=42, n_estimators=config["n_estimators"], class_weight=config["class_weight"])

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_val):
        return self.model.predict(X_val)
    
    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)[:, 1]
    
class MLModelTorch(nn.Module):
    def __init__(self):
        super().__init__()

    def validate(self, val_loader, criterion, device):
        print("DEBUG VALIDATE")
        predictions = []
        with torch.no_grad():
            self.eval()
            val_loss = 0.0
            for instances in val_loader:
                input = instances[0].to(device, dtype = torch.float32)
                labels = instances[1].to(device, dtype = torch.float32)
                labels = labels.unsqueeze(1)
                pred = self(input)
                loss = criterion(pred, labels)
                predictions.append(pred)
                val_loss += loss.item()
            
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss, predictions

    def fit(self, X_train, y_train, X_val, y_val):
        print("DEBUG FIT")
        y_val = torch.from_numpy(y_val).to(device, dtype = torch.float32)
        X_val = torch.from_numpy(X_val).to(device, dtype = torch.float32)
        y_train = torch.from_numpy(y_train).to(device, dtype = torch.float32)
        X_train = torch.from_numpy(X_train).to(device, dtype = torch.float32)

        num_positives = torch.sum(y_train, dim=0)
        num_negatives = len(y_train) - num_positives
        pos_weight  = num_negatives / num_positives

        epochs = 200
        patience = 5
        batch_size = 128

        optimizer = optim.Adam(self.parameters(), lr=0.0005)
        criterion = nn.BCEWithLogitsLoss(pos_weight)

        train_set_fold = TensorDataset(X_train, y_train)
        val_set_fold = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_set_fold, batch_size=batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_set_fold, batch_size=batch_size, num_workers=16)

        train_losses = []
        val_losses = []
        best_val_loss = np.inf

        counter = 0

        for epoch in range(epochs):
            print(epoch)
            self.train()
            train_loss = 0.0
            for batch_idx, instances in enumerate(train_loader):
                input = instances[0].to(device, dtype = torch.float32)
                labels = instances[1].to(device, dtype = torch.float32)
                labels = labels.unsqueeze(1)
                optimizer.zero_grad(set_to_none = True)
                predictions = self(input)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                    
            avg_train_loss = train_loss/len(train_loader)
            train_losses.append(avg_train_loss)

            avg_val_loss,_ = self.validate(val_loader, criterion, device)
            if avg_val_loss<=best_val_loss:
                counter = 0
                best_val_loss = avg_val_loss
            else:
                counter+=1
                if counter >= patience:
                    break

            val_losses.append(avg_val_loss)

        return train_losses, val_losses

    def predict_proba(self, X_val):
        print("DEBUG PREDICTPROBA")
        X_val = torch.from_numpy(X_val).to(device)
        with torch.no_grad():
            y_val = torch.sigmoid(self(X_val))
            return y_val.cpu().numpy()
        
    def forward(self, x):
        if hasattr(self, 'fc_output'):
            x = self.activation(self.fc_input(x))
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
            x = self.fc_output(x)
        else:
            x = self.fc_input(x)
        return x
        
    def predict(self, X_val):
        pass

class MLP(MLModelTorch):
    def __init__(self, config):
        super().__init__()
        self = self.to(device)
        if config.n_layers == 0:
            self.fc_input = nn.Linear(6000, 1)
        else:
            self.fc_input = nn.Linear(6000, config.hidden_dim)
            self.hidden_layers = nn.ModuleList([
                nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.n_layers - 1)
            ])
            self.fc_output = nn.Linear(config.hidden_dim, 1)
            if config.hidden_act == "leakyrelu":
                self.activation = nn.LeakyReLU()
            elif config.hidden_act == "sigmoid":
                self.activation = nn.Sigmoid()
            elif config.hidden_act == "tanh":
                self.activation = nn.Tanh()
            elif config.hidden_act == "relu":
                self.activation = nn.ReLU()
            elif config.hidden_act == "linear":
                self.activation = lambda x: x


