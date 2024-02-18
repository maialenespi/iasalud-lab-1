import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, record_function, ProfilerActivity
from torcheval.metrics.functional import r2_score
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import wandb


def get_datasets(batch_size):
    dataset_E_train, dataset_E_val, dataset_E_test, dataset_C_train, dataset_C_val, dataset_C_test = create_and_split_dataset()

    dataloader_E_train = DataLoader(dataset_E_train, batch_size=batch_size, shuffle=True, num_workers = 16)
    dataloader_E_val = DataLoader(dataset_E_val, batch_size=batch_size, shuffle=True, num_workers = 16)
    dataloader_E_test = DataLoader(dataset_E_test, batch_size=batch_size, shuffle=True, num_workers = 16)

    dataloader_C_train = DataLoader(dataset_C_train, batch_size=batch_size, shuffle=True, num_workers = 16)
    dataloader_C_val = DataLoader(dataset_C_val, batch_size=batch_size, shuffle=True, num_workers = 16)
    dataloader_C_test = DataLoader(dataset_C_test, batch_size=batch_size, shuffle=True, num_workers = 16)

    return dataloader_E_train, dataloader_E_val, dataloader_E_test, dataloader_C_train, dataloader_C_val, dataloader_C_test


def fit(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device, name):

    train_losses = []
    val_losses = []
    best_val_loss = np.inf

    scaler = GradScaler()
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 20, factor = 0.75, verbose=True)
    counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, instances in enumerate(train_loader):
            input = instances[0].to(device, dtype = torch.float32)
            if 'AE' not in name:
                labels = instances[1].to(device, dtype = torch.float32)
                labels = labels.unsqueeze(1)
            optimizer.zero_grad(set_to_none = True)
            
            if 'AE' not in name:
                predictions = model(input)
                loss = criterion(predictions, labels)
            else:
                _, predictions = model(input)
                loss = criterion(predictions, input)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            """
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] - Batch [{batch_idx + 1}/{len(train_loader)}] - "
                    f"Train Loss: {loss.item():.4f}")
            """
                
        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss,_ = validate(model, val_loader, criterion, device, name)
        if avg_val_loss<=best_val_loss:
            counter = 0
            torch.save(model.state_dict(), f"models/{name}.pth")
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
            if 'AE' not in name:
                labels = instances[1].to(device, dtype = torch.float32)
                labels = labels.unsqueeze(1)
            if 'AE' not in name:
                pred = model(input)
                loss = criterion(pred, labels)
            else:
                _, pred = model(input)
                loss = criterion(pred, input)
            predictions.append(pred)
            val_loss += loss.item()
        
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, predictions

def calcular_mape(y_true, y_pred):
    # Calcular el Error Porcentual Absoluto (APE) para cada muestra
    ape = torch.abs((y_true - y_pred) / y_true)
    # Calcular el Error Porcentual Absoluto Medio (MAPE)
    mape = torch.mean(ape) * 100
    return mape.item()

def test_mape(model, test_loader, device):
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        mape_total = 0.0
        total_samples = 0
        for inputs in test_loader:
            inputs = inputs[0].to(device, dtype = torch.float32)  # Mover los datos a la GPU si está disponible

            # Realizar predicciones
            with autocast(dtype=torch.float32):
                _, outputs = model(inputs)

            # Calcular el MAPE para el lote actual
            mape_batch = calcular_mape(inputs, outputs)
            print(mape_batch)
            # Actualizar el MAPE total y el número total de muestras
            mape_total += mape_batch
            total_samples += inputs.size(0)

# Calcular el MAPE promedio en el conjunto de prueba
    mape_promedio = mape_total / total_samples
    return mape_promedio

def test(model, test_loader, criterion, device, name):
    test_loss, predictions = validate(model, test_loader, criterion, device, name)
    if 'AE' in name:
        mape = test_mape(model, test_loader, device)
        wandb.run.summary["test_mape"] =  mape
        print("Test mape: ", mape)

    #wandb.run.summary["test_loss"] =  test_loss
    #print("Test loss: ", test_loss)

    real_labels = []
    if 'AE' not in name:
        for input, labels in test_loader:
            real_labels.extend(labels)

    if 'AE' not in name:
        t = []
        for i in predictions:
            i = torch.sigmoid(i)
            t.extend(i.cpu())
        t = np.array([i.item() for i in t])
        for i in range(t.shape[0]):
            if t[i]>=0.5:
                t[i] = 1.0
            else:
                t[i] = 0.0

    #if 'AE' not in name:
        #print("Predictions: ", [f"{pred:.2f}" for pred in t[:10]])
        #print("Real labels: ", [pred.item() for pred in real_labels[:10]])
    
def plot_metrics(metrics, name):
    for metric in metrics:
        plt.plot(metric)
    plt.savefig(f"./results/{name}.png")
    plt.clf()  # Clear the current figure
    plt.close()