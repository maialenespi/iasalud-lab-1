import wandb
import yaml
import torch
from preprocess import preprocess_data, reduce_dimensionality
from dataset_mng import create_dataset, split_dataset
from train import train_model

def initialize_wandb(project_name, entity_name, name, config):
    wandb.init(project=project_name, entity=entity_name, name=name, config=config)

def main(config_file, data):
    torch.manual_seed(0)
    # Cargar la configuración de la pipeline desde el archivo
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Inicializar wandb
    #initialize_wandb(config['project'], config['entity'], config['name'], config)

    prep = {'func': config['preprocessing']}
    reduc = {'func': config['reduc_func'],
             'params': config[config['reduc_func']]}
    model = {'func': config['model_function'],
             'params': config[config['model_function']]}
    validation = {'func': config['validation']}
    
    print(prep, reduc, model)
    # Ejecutar la pipeline con la configuración especificada
    metrics = run_pipeline(data, prep, reduc, model, validation)
    results = {'Modelo Erythromycin': {
                'accuracy': metrics[0][0],
                'precision': metrics[0][1],
                'recall': metrics[0][2],
                'pr_auc': metrics[0][3],
                'roc_auc': metrics[0][4]
                },
                'Modelo Ciprofloxacin': {
                'accuracy': metrics[1][0],
                'precision': metrics[1][1],
                'recall': metrics[1][2],
                'pr_auc': metrics[1][3],
                'roc_auc': metrics[1][4]}
                }
    # Guardar los resultados en wandb
    for modelo in results.keys():
        for metric in results[modelo].keys():
            print(f"{modelo}_{metric}", results[modelo][metric])
            #wandb.run.summary[f"{modelo}_{metric}"] = results[modelo][metric]
    #wandb.log(results)
    auc_mean = (results['Modelo Erythromycin']['auc'] + results['Modelo Ciprofloxacin']['auc'])/2
    #wandb.log({'auc': auc_mean})
    #wandb.save(config_file)
    #print(results)

def run_pipeline(data, preprocessing, reduction, model, validation):
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

    tr_params = {'weight': 'yes',
                'l2': 0.02,
                'resampling': 'random'}

    metrics_E = train_model(X, Y_E, model_func, model_params, tr_params, validation_func, 'E')
    metrics_C = train_model(X, Y_C, model_func, model_params, tr_params, validation_func, 'C')

    return metrics_E, metrics_C
    

if __name__ == "__main__":
    df = create_dataset()
    main('configs/pipeline_config.yaml', df)