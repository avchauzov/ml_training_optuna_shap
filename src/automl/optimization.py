"""
This script provides functions for automating the machine learning model selection, hyperparameter optimization, and feature selection process using Optuna.
"""
import time
import warnings

import numpy as np
import optuna

from src.automl.metric_functions import METRIC_FUNCTIONS
from src.automl.study import optuna_hyperparameter_optimization
from src.models.models import MODELS
from src.scripts.feature_selection import feature_selection
from src.utils.tasks import TASKS


warnings.filterwarnings('always')
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def validate_task_model_metric(task_name, model_name, metric_name):
	"""
	Validates the input task name, model name, and metric name to ensure they are valid.

	Args:
		task_name (str): The name of the task.
		model_name (str): The name of the machine learning model.
		metric_name (str): The name of the evaluation metric.

	Raises:
		ValueError: If any of the input names are not recognized.

	"""
	if task_name not in TASKS:
		raise ValueError(f'Error: Unknown task name: {task_name}')
	
	if model_name not in MODELS:
		raise ValueError(f'Error: Unknown model name: {model_name}')
	
	if metric_name not in METRIC_FUNCTIONS:
		raise ValueError(f'Error: Unknown metric name: {metric_name}')


def find_best_model(task_name, model_name, metric_name, x_data, y_data, weight_data, cv, n_trials_long, n_trials_short, patience=0.1, n_jobs=-1, drop_rate=0.5, min_columns_to_keep=8, test_mode=False):
	"""
	Automates the process of model selection, hyperparameter optimization, and feature selection.

	Args:
		task_name (str): The name of the task.
		model_name (str): The name of the machine learning model.
		metric_name (str): The name of the evaluation metric.
		x_data (array-like): The feature data.
		y_data (array-like): The target data.
		weight_data (array-like, optional): The sample weight data. Default is None.
		cv (list): List of cross-validation fold indices.
		n_trials_long (int): Number of trials for long hyperparameter optimization.
		n_trials_short (int): Number of trials for short hyperparameter optimization.
		patience (int): The patience parameter for early stopping during optimization.
		n_jobs (int): Number of parallel jobs for optimization.
		drop_rate (float): Rate of columns to drop in feature selection. Default is 0.50.
		min_columns_to_keep (int): Minimum number of columns to keep in feature selection. Default is 8.
		test_mode (bool): Whether to run in test mode. Default is False.

	Returns:
		tuple: A tuple containing the best hyperparameters and important features.

	"""
	validate_task_model_metric(task_name, model_name, metric_name)
	
	if weight_data is None:
		weight_data = np.ones(len(y_data))
	elif not isinstance(weight_data, np.ndarray):
		weight_data = np.array(weight_data)
	
	warnings_list = []
	
	print('Starting optimization with the following parameters:')
	print(f'task_name="{task_name}", model_name="{model_name}", metric_name="{metric_name}", n_trials_long={n_trials_long}, n_trials_short={n_trials_short}, patience={patience}, n_jobs={n_jobs}, drop_rate={drop_rate}, min_columns_to_keep={min_columns_to_keep}')
	print()
	
	# Step 1: Hyperparameters Optimization
	print('Step 1: Hyperparameters Optimization')
	time.sleep(1)
	
	with warnings.catch_warnings(record=True) as _warnings:
		best_hyperparameters = optimize_hyperparameters([x_data, y_data, weight_data], {}, cv, n_trials_long, patience, 'long', n_jobs, task_name, model_name, metric_name, test_mode)
		warnings_list.extend([(str(warning.category), str(warning.message)) for warning in _warnings])
	
	print(list(set(warnings_list)))
	
	if (task_name, model_name) == ('classification_multiclass', 'multinomialnb'):
		return best_hyperparameters, list(x_data)
	
	time.sleep(1)
	print('\n')
	
	# Step 2: Feature selection
	print('Step 2: Feature Selection')
	time.sleep(1)
	
	with warnings.catch_warnings(record=True) as _warnings:
		important_features_list = feature_selection([x_data, y_data, weight_data], cv, best_hyperparameters, drop_rate, min_columns_to_keep, task_name, model_name, metric_name)
		warnings_list.extend([(str(warning.category), str(warning.message)) for warning in _warnings])
	
	print(list(set(warnings_list)))
	
	time.sleep(1)
	print('\n')
	
	# Step 3: Hyperparameters Fine-Tuning
	print('Step 3: Hyperparameters Fine-Tuning')
	time.sleep(1)
	
	with warnings.catch_warnings(record=True) as _warnings:
		best_hyperparameters = optimize_hyperparameters([x_data[important_features_list], y_data, weight_data], best_hyperparameters, cv, n_trials_short, patience, 'short', n_jobs, task_name, model_name, metric_name, test_mode)
		warnings_list.extend([(str(warning.category), str(warning.message)) for warning in _warnings])
	
	print(list(set(warnings_list)))
	
	time.sleep(1)
	print('\n')
	
	return best_hyperparameters, important_features_list


def optimize_hyperparameters(data, hyperparameters, cv, trials, patience, trial_type, n_jobs, task_name, model_name, metric_name, test_mode):
	"""
	Optimizes hyperparameters using Optuna for a given data and hyperparameter search space.

	Args:
		data (list): List containing x_data, y_data, and sample_weight_data.
		hyperparameters (dict): The hyperparameter search space.
		cv (list): List of cross-validation fold indices.
		trials (int): Number of trials for optimization.
		patience (int): The patience parameter for early stopping during optimization.
		trial_type (str): The type of optimization (e.g., 'long' or 'short').
		n_jobs (int): Number of parallel jobs for optimization.
		task_name (str): The name of the task.
		model_name (str): The name of the machine learning model.
		metric_name (str): The name of the evaluation metric.
		test_mode (bool): Whether to run in test mode.

	Returns:
		dict: The best hyperparameters.

	Raises:
		Exception: If no valid hyperparameters are found during optimization.

	"""
	best_hyperparameters = optuna_hyperparameter_optimization(data, cv, hyperparameters, trials, int(np.ceil(patience * trials)), trial_type, n_jobs, task_name, model_name, metric_name, test_mode)
	
	if not best_hyperparameters:
		raise Exception('Hyperparameter optimization did not find any valid hyperparameters. Please check your input data, hyperparameter search space, or "n_trials" value.')
	
	return best_hyperparameters
