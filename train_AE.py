import numpy as np
from train_models import train_model, plot_metrics, test
from torch.utils.data import TensorDataset, DataLoader
from utils import create_dataset, preprocess_data
import torch
from models import AutoEncoder
from torch.utils.data.dataset import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import argparse

wandb.init(project="lab-1", entity="reeses", name = f"AE training")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AE training')
    parser.add_argument('--latent_dim', '-lt', type=int, required=True, help="Dimension latent space")
    parser.add_argument('--scaler', '-s', type=str, required=True, help="Scaler function")
    args = parser.parse_args()

    dataset = create_dataset()
    dataset = preprocess_data(dataset, args.scaler)
    X = dataset.drop(["Erythromycin","Ciprofloxacin"], axis=1).values
    X = torch.from_numpy(X)
    dataset = TensorDataset(X)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    batch_size = 64

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 16)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers = 16)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers = 16)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    model = AutoEncoder(latent_dim=args.latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    epochs = 200
    patience = 5

    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device, f'AE_{args.latent_dim}_{args.scaler}')
    plot_metrics((train_losses, val_losses), f'AE_{args.latent_dim}_{args.scaler}')
    model.load_state_dict(torch.load(f"models/AE_{args.latent_dim}_{args.scaler}.pth"))
    test(model, test_loader, criterion, device, f'AE_{args.latent_dim}_{args.scaler}')

    wandb.run.summary['latent dimension'] = args.latent_dim
    wandb.run.summary['scaler'] = args.scaler
    wandb.finish()
