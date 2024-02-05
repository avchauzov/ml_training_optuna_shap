import numpy as np
import optuna

from automl.settings import METRICS, MODELS, TASKS
from optimization.optimization import optuna_optimization
from scripts.feature_selection import get_select_features


optuna.logging.set_verbosity(optuna.logging.INFO)


def find_best_model(task_name, model_name, metric_name, x_data, y_data, weight_data, cv, n_trials_long, n_trials_short, patience, n_jobs, drop_rate=0.50, min_columns_to_keep=8, test_mode=False):
	if task_name not in TASKS:
		raise ValueError(f'Error: Unknown task name: {task_name}')
	
	if model_name not in MODELS:
		raise ValueError(f'Error: Unknown model name: {model_name}')
	
	if metric_name not in list(METRICS.keys()):
		raise ValueError(f'Error: Unknown metric name: {metric_name}')
	
	if weight_data is None:
		weight_data = np.ones(len(y_data))
	
	elif not isinstance(weight_data, np.ndarray):
		weight_data = np.array(weight_data)
	
	print('Step 1: Hyperparameters General Optimization')
	best_hyperparameters = optuna_optimization([x_data, y_data, weight_data], cv, {}, n_trials_long, patience, 'long', n_jobs, task_name, model_name, metric_name, test_mode)
	
	if best_hyperparameters == {}:
		raise Exception('Hyperparameter optimization did not find any valid hyperparameters. Please check your input data, hyperparameter search space or "n_trials" value.')
	
	if task_name == 'classification_multiclass' and model_name == 'multinomialnb':
		return best_hyperparameters, list(x_data)
	
	return best_hyperparameters, list(x_data)
	
	print('Step 2: Feature selection')
	important_features_list = get_select_features(
			x_data, y_data, weight_data, cv, best_hyperparameters, weight_adjustment, drop_rate, min_columns_to_keep,
			scoring, task_name, model_name, metric_dictionary
			)
	
	print('Step 3: learning_rate and n_estimators optimization')
	best_hyperparameters['weight_adjustment'] = weight_adjustment
	best_hyperparameters, _ = optuna_optimization(x_data[important_features_list], y_data, weight_data, best_hyperparameters, n_jobs, patience, cv, scoring, 'short', n_trials_short, task_name)
	
	return best_hyperparameters, weight_adjustment, important_features_list
