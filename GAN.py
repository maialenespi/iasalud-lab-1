import pandas as pd
from dataset_mng import create_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from train_models import plot_metrics
from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from models import MLP
from sklearn.neighbors import KernelDensity
from scipy.special import kl_div
import joblib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, lt):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            
            nn.Linear(lt, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 6000),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(6000, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def kl_divergence(mu_fake, logvar_fake, mu_real, logvar_real):
    kl_loss = -0.5 * torch.sum(1 + logvar_fake - logvar_real - (logvar_fake.exp() + (mu_fake - mu_real).pow(2)) / logvar_real.exp(), dim=0)
    return kl_loss.mean()

def train(dataloader, type):
    
    lt = 100
    lr = 0.0001
    n_epochs = 500

    netD = Discriminator().to(device)
    netG = Generator(lt).to(device)
    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    #fixed_noise = torch.randn(64, lt, 1, 1, device=device)


    img_list = []
    G_losses = []
    D_losses = []
    D_real_l = []
    D_fake_l = []
    for epoch in range(n_epochs):
        G_loss_ep = []
        D_loss_ep = []
        D_real_l_ep = []
        D_fake_l_ep = []
        for i, data in enumerate(dataloader, 0):
            #train D with real data
            netD.zero_grad()
            data = data[0].to(device, dtype = torch.float)
            b_size = data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.cpu().mean().item()

            #train D with fake data
            noise = torch.randn(b_size, lt, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            #update D
            errD = errD_real + errD_fake
            optimizerD.step()

            
            #update G based on how well it fools D
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)

            """
            real_data_prob = nn.functional.softmax(data, dim=1)
            fake_data_prob = nn.functional.softmax(fake, dim=1)

            # Luego, calcula las log-probabilidades de real_data
            real_data_log_prob = torch.log(real_data_prob)

            # Calcula la divergencia KL
            kl_div = nn.functional.kl_div(real_data_log_prob, fake_data_prob, reduction='batchmean')
            #print(kl_div)
            """

            errG = criterion(output, label) #+ torch.sqrt(nn.functional.mse_loss(fake, data))
            errG.backward()
            #D_G_z2 = output.mean().item()
            optimizerG.step()


            G_loss_ep.append(errG.item())
            D_loss_ep.append(errD.item())
            D_real_l_ep.append(D_x)
            D_fake_l_ep.append(D_G_z1)

        G_losses.append(np.mean(G_loss_ep))
        D_losses.append(np.mean(D_loss_ep))
        D_real_l.append(np.mean(D_real_l_ep))
        D_fake_l.append(np.mean(D_fake_l_ep))
        #fake = netG(fixed_noise).detach().cpu()
        print(f"Epoch [{epoch + 1}/{n_epochs}] - D loss: {D_losses[-1]:.4f}, G Loss: {G_losses[-1]:.4f}- ", end='')
        print(f"D on real data: {D_real_l[-1]:.4f} - D on fake data: {D_fake_l[-1]:.4f}\n")

        if epoch>100 and D_real_l[-1] >0.497 and D_real_l[-1]<0.503 and D_fake_l[-1]>0.497 and D_fake_l[-1]<0.503:
            print("Converged")
            break

    torch.save(netD.state_dict(), f"models/GAN_D_{type}.pth")
    torch.save(netG.state_dict(), f"models/GAN_G_{type}.pth")
    plot(G_losses, D_losses, D_real_l, D_fake_l, type)
    return (G_losses, D_losses, D_real_l, D_fake_l)
    
def plot(G_losses, D_losses, D_real_l, D_fake_l, t):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="Generator")
    plt.plot(D_losses,label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/GAN_errors_{t}.png")

    plt.figure(figsize=(10,5))
    plt.title("Discriminator Output Probabilities During Training")
    plt.plot(D_real_l,label="Real Data")
    plt.plot(D_fake_l,label="Fake Data")
    plt.xlabel("Epoch")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(f"results/GAN_D_{t}.png")

if __name__ == "__main__":
    df = create_dataset()
    df =  preprocess_data(df, 'normalizer_l2')
    X_E = df[df['Erythromycin'] == 1].iloc[:, :-2].values
    X_C = df[df['Ciprofloxacin'] == 1].iloc[:, :-2].values

    #print(np.std(X_E))
    X_E = torch.from_numpy(X_E)
    X_C = torch.from_numpy(X_C)
    
    real_E = X_E[np.random.randint(100)].cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.plot(real_E)
    plt.savefig("real_E.png")

    real_C = X_C[np.random.randint(100)].cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.plot(real_C)
    plt.savefig("real_C.png")
    
    #print(torch.std(X_E))
    print(X_E.size(), X_C.size(), torch.max(X_E), torch.min(X_C))

    dataset_E_train = TensorDataset(X_E)
    dataset_E_test = TensorDataset(X_E)
    dataset_C_train = TensorDataset(X_C)
    dataset_C_test = TensorDataset(X_C)

    batch_size = 64
    train_loader_E = DataLoader(dataset_E_train, batch_size=batch_size, shuffle=True,num_workers = 8)
    test_loader_E = DataLoader(dataset_E_test, batch_size=batch_size, shuffle=False)
    train_loader_C = DataLoader(dataset_C_train, batch_size=batch_size, shuffle=True, num_workers = 8)
    test_loader_C = DataLoader(dataset_C_test, batch_size=batch_size, shuffle=False)

    train(train_loader_E, 'E')
    train(train_loader_C, 'C')

    lt = 100
    G_E = Generator(lt).to(device)
    G_E.load_state_dict(torch.load(f"models/GAN_G_E.pth"))
    G_E.eval()

    G_C = Generator(lt).to(device)
    G_C.load_state_dict(torch.load(f"models/GAN_G_C.pth"))
    G_C.eval()

    noise = torch.randn(1, lt, device=device)
    with torch.no_grad():
        gen_E = G_E(noise)
        gen_C = G_C(noise)

    gen_E = gen_E.view(-1).cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.plot(gen_E)
    plt.savefig("generated_E.png")

    gen_C = gen_C.view(-1).cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.plot(gen_C)
    plt.savefig("generated_C.png")


    noise_E = torch.randn(X_E.size(0), lt, device=device)
    noise_C = torch.randn(X_C.size(0), lt, device=device)
    with torch.no_grad():
        gen_E = G_E(noise_E)
        gen_C = G_C(noise_C)

    gen_E = normalize(gen_E.cpu().numpy(), norm='l2')
    gen_C = normalize(gen_C.cpu().numpy(), norm='l2')
    joblib.dump(gen_E, 'fake_E.pkl')
    joblib.dump(gen_C, 'fake_C.pkl')
