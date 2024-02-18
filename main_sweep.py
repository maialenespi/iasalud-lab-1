from preprocess import preprocess_data, reduce_dimensionality
from dataset_mng import create_dataset, split_dataset
from train import train_model
import wandb
import yaml
import torch

def initialize_wandb(sweep_config):
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=main)

def main():
    run = wandb.init()
    config = run.config
    torch.manual_seed(0)

    prep = {'func': config.preprocessing}

    reduc = {'func': 'None',
            'params': {}}
    
    if model_func=='mlp':
        model = {'func': model_func,
                'params': {'hidden_size': config.hidden_size,
                            'num_layers': config.num_layers}
                }
    else:
        model = {'func': model_func,
                'params': {'hidden_size': config.hidden_size,
                            'num_layers': config.num_layers,
                            'kernel_size': config.kernel_size,
                            'stride': config.stride}
                }
    
    tr_params = {'weight': config.weight,
                'l2': config.l2}
    
    validation = {'func': "kfolds"}

    print(prep, reduc, model, tr_params)
    # Ejecutar la pipeline con la configuración especificada
    data = create_dataset()
    metrics = run_pipeline(data, prep, reduc, model, tr_params, validation)
    results = {
                'accuracy': metrics[0],
                'precision': metrics[1],
                'auc': metrics[2]
                }
    # Guardar los resultados en wandb
    for metric in results.keys():
        #print(f"{metric}", results[metric])
        wandb.run.summary[f"{metric}"] = results[metric]
    #wandb.log(results)
    wandb.log({'auc': results['auc']})
    wandb.log({'precision': results['precision']})
    wandb.log({'accuracy': results['accuracy']})
    wandb.save(config_file)
    #print(results)

def run_pipeline(data, preprocessing, reduction, model, tr_params, validation):
    prep_func = preprocessing['func']
    red_func = reduction['func']
    red_params = reduction['params']
    model_func = model['func']
    model_params = model['params']
    validation_func = validation['func']

    data = preprocess_data(data, prep_func)
    data = reduce_dimensionality(data, red_func, red_params)

    if validation_func == 'split_test':
        X, Y = split_dataset(data)
        Y_E, Y_C = Y
    else:
        X = data.drop(["Erythromycin","Ciprofloxacin"], axis=1)
        Y_E = data["Erythromycin"]
        Y_C = data["Ciprofloxacin"]


    if t=='E':
        metrics = train_model(X, Y_E, model_func, model_params, tr_params, validation_func, t)
    else:
        metrics = train_model(X, Y_C, model_func, model_params, tr_params, validation_func, t)

    return metrics
    

if __name__ == "__main__":
    config_file = 'configs/sweep_cnn_E.yaml'
    with open(config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)

    global model_func, t
    t = config_file[:-6]
    model_func = config_file.split('configs/sweep_')[-1][:3]
    initialize_wandb(sweep_config)