from sklearn.linear_model import LogisticRegression as LRBase
from sklearn.ensemble import RandomForestClassifier as RFBase
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, f1_score, average_precision_score
from tensorflow.keras import models
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf

class MLModel:
    def __init__(self, **config):
        pass
    
    def fit(self, X_train, y_train):
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

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_val):
        return self.model.predict(X_val)
    
    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)[:, 1]
    
class RandomForest(MLModel):
    def __init__(self, config):
        super().__init__()
        self.model = RFBase(random_state=42, n_estimators=config["n_estimators"], class_weight=config["class_weight"])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_val):
        return self.model.predict(X_val)
    
    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)[:, 1]
    
class MLP_TF(MLModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def fit(self, X_train, y_train):
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

    def predict(self, X_val):
        return tf.round(self.model.predict(X_val))
    
    def predict_proba(self, X_val):
        return self.model.predict(X_val)
    
class CNN_TF(MLModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def fit(self, X_train, y_train):
        model = models.Sequential()
        model.add(Conv1D(self.config["hidden_dim"], kernel_size=self.config["kernel_size"], activation=self.config["hidden_act"], input_shape=X_train.shape))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        class_weight = {}
        class_weight[0], class_weight[1]  = compute_class_weight(class_weight=self.config["class_weight"], classes=np.unique(y_train), y=y_train)
        self.model.fit(X_train, y_train, epochs=100, class_weight=class_weight, verbose=0)
    
    def predict(self, X_val):
        return tf.round(self.model.predict(X_val))
    
    def predict_proba(self, X_val):
        return self.model.predict(X_val)