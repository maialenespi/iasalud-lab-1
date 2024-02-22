from sklearn.linear_model import LogisticRegression as LRBase
from sklearn.ensemble import RandomForestClassifier as RFBase
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, f1_score, average_precision_score
#from tensorflow.keras import models
#from tensorflow.keras.layers import Conv1D, Flatten, Dense
#from sklearn.utils.class_weight import compute_class_weight
import numpy as np
#import tensorflow as tf
#from tensorflow.keras.callbacks import EarlyStopping
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MLModel:
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

class LogisticRegression(MLModel):
    def __init__(self, config):
        super().__init__()
        self.model = LRBase(random_state=42, penalty=config['penalty'], class_weight=config['class_weight'])

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_val):
        return self.model.predict(X_val)
    
    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)[:, 1]
    
class RandomForest(MLModel):
    def __init__(self, config):
        super().__init__()
        self.model = RFBase(random_state=42, n_estimators=config["n_estimators"], class_weight=config["class_weight"])

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_val):
        return self.model.predict(X_val)
    
    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)[:, 1]

"""
class MLP_TF(MLModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def fit(self, X_train, y_train, X_val, y_val):
        model = models.Sequential()
        model.add(Dense(self.config["hidden_dim"], activation=self.config["hidden_act"], input_dim=X_train.shape[1]))
        for _ in range(self.config["n_layers"]):
            model.add(Dense(self.config["hidden_dim"], activation=self.config["hidden_act"]))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        class_weight = {}
        class_weight[0], class_weight[1]  = compute_class_weight(class_weight=self.config["class_weight"], classes=np.unique(y_train), y=y_train)
        self.model.fit(X_train, y_train, epochs=100, class_weight=class_weight, verbose=0)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), class_weight=class_weight, verbose=0, callbacks=[early_stopping])

    def predict(self, X_val):
        return tf.round(self.model.predict(X_val))
    
    def predict_proba(self, X_val):
        return self.model.predict(X_val)

class CNN_TF(MLModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def fit(self, X_train, y_train, X_val, y_val):
        model = models.Sequential()
        model.add(Conv1D(self.config["hidden_dim"], kernel_size=self.config["kernel_size"], activation=self.config["hidden_act"], input_shape=X_train.shape))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        class_weight = {}
        class_weight[0], class_weight[1]  = compute_class_weight(class_weight=self.config["class_weight"], classes=np.unique(y_train), y=y_train)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), class_weight=class_weight, verbose=0, callbacks=[early_stopping])
    
    def predict(self, X_val):
        return tf.round(self.model.predict(X_val))
    
    def predict_proba(self, X_val):
        return self.model.predict(X_val)
"""
        
        
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation):
        super(MLP, self).__init__()

        if num_layers == 0:
            self.fc_input = nn.Linear(input_size, output_size)
        else:
            self.fc_input = nn.Linear(input_size, hidden_size)
            self.hidden_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
            ])
            self.fc_output = nn.Linear(hidden_size, output_size)
            if activation == "leakyrelu":
                self.activation = nn.LeakyReLU()
            elif activation == "sigmoid":
                self.activation = nn.Sigmoid()
            elif activation == "tanh":
                self.activation = nn.Tanh()
            elif activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "linear":
                self.activation = lambda x: x
            
    def forward(self, x):
        if hasattr(self, 'fc_output'):  # Si hay capas ocultas
            x = self.activation(self.fc_input(x))
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
            x = self.fc_output(x)
        else:  # Si no hay capas ocultas
            x = self.fc_input(x)
        return x

class Pytorch_Model(MLModel):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def fit(self, X_train, y_train, X_val, y_val):
        model = self.model(6000, self.config.hidden_dim, self.config.n_layers, 1, self.config.hidden_act)
        
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

        optimizer = optim.Adam(model.parameters(), lr=0.0005)
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
            model.train()
            train_loss = 0.0
            for batch_idx, instances in enumerate(train_loader):
                input = instances[0].to(device, dtype = torch.float32)
                labels = instances[1].to(device, dtype = torch.float32)
                labels = labels.unsqueeze(1)
                optimizer.zero_grad(set_to_none = True)
                predictions = model(input)
                loss = criterion(predictions, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                    
            avg_train_loss = train_loss/len(train_loader)
            train_losses.append(avg_train_loss)

            avg_val_loss,_ = self.validate(model, val_loader, criterion, device)
            if avg_val_loss<=best_val_loss:
                counter = 0
                #torch.save(model.state_dict(), f"models/{name}.pth")
                best_val_loss = avg_val_loss
            else:
                counter+=1
                if counter >= patience:
                    #print("Deteniendo el entrenamiento debido a early stopping.")
                    break

            #scheduler.step(avg_val_loss)

            val_losses.append(avg_val_loss)
            #print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n")
            #wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch+1})

        return train_losses, val_losses

    def validate(model, val_loader, criterion, device, name):
        predictions = []
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for instances in val_loader:
                input = instances[0].to(device, dtype = torch.float32)
                labels = instances[1].to(device, dtype = torch.float32)
                labels = labels.unsqueeze(1)
                pred = model(input)
                loss = criterion(pred, labels)
                loss = criterion(pred, input)
                predictions.append(pred)
                val_loss += loss.item()
            
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss, predictions


