from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import pandas as pd
import matplotlib.pyplot as plt
import torch
from models import AutoEncoder
import wandb


def preprocess_data(data, preprocessing):
    Y_E, Y_C = data['Erythromycin'], data['Ciprofloxacin']
    data = data.drop(["Erythromycin","Ciprofloxacin"], axis=1)

    if preprocessing == 'minmaxscaler':
        scaler = MinMaxScaler()
    elif preprocessing == 'standarizer':
        scaler = StandardScaler()
    elif 'normalizer' in preprocessing:
        norm = preprocessing.split('_')[-1]
        data = normalize(data, norm)
        data = pd.DataFrame(data)
        data = pd.concat([data, Y_E, Y_C], axis=1)
        return data
    else:
        data = pd.concat([data, Y_E, Y_C], axis=1)
        return data
    data = scaler.fit_transform(data)

    data = pd.DataFrame(data)
    data = pd.concat([data, Y_E, Y_C], axis=1)
    return data

def reduce_dimensionality(data, reduction, model_params):
    if reduction == 'None':
        return data
    
    Y_E, Y_C = data['Erythromycin'], data['Ciprofloxacin']
    data = data.drop(["Erythromycin","Ciprofloxacin"], axis=1)
    
    if reduction == 'pca':
        data = apply_pca(data, model_params)

    elif reduction == 'tsne':
        data = apply_tsne(data, model_params)

    elif reduction == 'umap':
        data = apply_umap(data, model_params)

    elif reduction == 'ae':
        data = apply_autoencoder(data, model_params)

    # Plot de los datos reducidos
    if reduction != 'none' and model_params['n_components'] == 2:
        plot_reduced_data(data, Y_E, reduction)

    data = pd.DataFrame(data)
    data = pd.concat([data, Y_E, Y_C], axis=1)
    return data

def apply_pca(data, model_params):
    pca = PCA(**model_params)
    data_reduced = pca.fit_transform(data)
    explained_variance_per_feature = pca.explained_variance_ratio_
    feature_names = [f"feature{i+1}" for i in range(model_params['n_components'])]  # Replace with your feature names

    # Plotting the variance explained by each feature
    plt.bar(feature_names, explained_variance_per_feature)
    plt.xlabel('Features')
    plt.ylabel('Variance Explained')
    plt.title('Variance Explained by Each Feature')
    plt.savefig('PCA.png')

    return data_reduced

def apply_tsne(data, model_params):
    model_params['n_jobs'] = 4
    model_params['perplexity'] = 30
    data_reduced = TSNE(**model_params).fit_transform(data)
    return data_reduced

def apply_umap(data, model_params):
    model_params['metric'] = 'euclidean'
    data_reduced = umap.UMAP(**model_params).fit_transform(data)
    return data_reduced

def apply_autoencoder(data, model_params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(data.values).to(device, torch.float32)
    model = AutoEncoder(model_params['n_components']).to(device)
    model.load_state_dict(torch.load(f"models/AE_{model_params['n_components']}_{model_params['preprocessing']}.pth"))
    model.eval()
    with torch.no_grad():
        data_reduced, _ = model(data_tensor)
    data_reduced = data_reduced.cpu().numpy()
    return data_reduced

def plot_reduced_data(data_reduced, Y_E, reduction):
    plt.figure(figsize=(8, 6))
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=Y_E, cmap='viridis', alpha=0.5)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Visualización de Datos Reducidos ({})'.format(reduction))
    plt.colorbar(label='Clase')
    plt.grid(False)
    plt.savefig(f"imagenes/{reduction}")
    image_path = f"imagenes/{reduction}.png"
    wandb.log({"image": wandb.Image(image_path)})