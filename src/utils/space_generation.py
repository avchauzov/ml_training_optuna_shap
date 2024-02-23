"""
This module contains functions for generating hyperparameters for different tasks, models, and optimization types.
"""
import json

import numpy as np


def elasticnet_long_parameters(trial):
	return load_optuna_parameters('src/_settings/optimization_hyperparameters/elasticnet.json', 'long', trial)


def elasticnet_short_parameters(trial):
	return load_optuna_parameters('src/_settings/optimization_hyperparameters/elasticnet.json', 'short', trial)


def lightgbm_long_parameters(trial, objective, metric, num_class, n_jobs):
	"""
	Generate LightGBM parameters for the 'long' configuration based on trial suggestions.

	Args:
		trial: Trial object for optimization.
		objective (list): List of objective functions.
		metric (list): List of evaluation metrics.
		num_class (int): Number of classes for multiclass tasks.
		n_jobs (int): Number of parallel jobs.

	Returns:
		dict: Dictionary of LightGBM hyperparameters.
	"""
	parameters = load_optuna_parameters('src/_settings/optimization_hyperparameters/lightgbm.json', 'long', trial)
	
	parameters.update(
			{
					'objective': trial.suggest_categorical('objective', objective),
					'metric'   : trial.suggest_categorical('metric', metric),
					'n_jobs'   : trial.suggest_categorical('n_jobs', [n_jobs])
					}
			)
	
	if num_class > 1:
		parameters.update({'num_class': trial.suggest_categorical('num_class', [num_class])})
	
	return parameters


# Function to generate LightGBM parameters for the 'short' configuration
def lightgbm_short_parameters(trial):
	"""
	Generate LightGBM parameters for the 'short' configuration based on trial suggestions.

	Args:
		trial: Trial object for optimization.

	Returns:
		dict: Dictionary of LightGBM hyperparameters.
	"""
	return load_optuna_parameters('src/_settings/optimization_hyperparameters/lightgbm.json', 'short', trial)

def load_optuna_parameters(file_path, _type, trial):
	"""
	Load parameters from a JSON file and convert them into Optuna suggest functions.

	Args:
		file_path (str): Path to the JSON file containing the parameters.

	Returns:
		dict: A dictionary containing Optuna suggest functions.
	"""
	with open(file_path) as file:
		optuna_parameters = json.load(file)
	
	optuna_parameters = optuna_parameters.get(_type, {})
	
	parameters = {}
	for key, value in optuna_parameters.items():
		column_type = value[0]
		
		if column_type in ['int']:
			_, min_value, max_value, step_log = value
			
			if isinstance(step_log, int):
				max_value = min_value + step_log * np.floor((max_value - min_value) / step_log)
				parameters[key] = trial.suggest_int(key, min_value, max_value, step=step_log)
			
			elif isinstance(step_log, bool):
				parameters[key] = trial.suggest_int(key, min_value, max_value, log=step_log)
			
			else:
				raise ''
		
		elif column_type in ['categorical']:
			_, values = value
			parameters[key] = trial.suggest_categorical(key, values)
		
		elif column_type in ['float']:
			_, min_value, max_value, step_log = value
			
			if isinstance(step_log, float):
				max_value = min_value + step_log * np.floor((max_value - min_value) / step_log)
				parameters[key] = trial.suggest_float(key, min_value, max_value, step=step_log)
			
			elif isinstance(step_log, bool):
				parameters[key] = trial.suggest_float(key, min_value, max_value, log=step_log)
		
		else:
			raise Exception(f'Unknown type: {column_type}')
	
	return parameters


def logisticregression_long_parameters(trial, n_jobs):
	parameters = load_optuna_parameters('src/_settings/optimization_hyperparameters/logisticregression.json', 'long', trial)
	parameters['n_jobs'] = trial.suggest_categorical('n_jobs', [n_jobs])
	
	return parameters


def logisticregression_short_parameters(trial):
	return load_optuna_parameters('src/_settings/optimization_hyperparameters/logisticregression.json', 'short', trial)


# Function to generate parameters for MultinomialNB
def multinomialnb_parameters(trial):
	"""
	Generate parameters for MultinomialNB based on trial suggestions.

	Args:
		trial: Trial object for optimization.

	Returns:
		dict: Dictionary of MultinomialNB hyperparameters.
	"""
	return load_optuna_parameters('src/_settings/optimization_hyperparameters/multinomialnb.json', 'long', trial)


# Function to set LightGBM production mode parameters based on trial suggestions
def set_lightgbm_production_mode_parameters(parameters, trial):
	"""
	Set LightGBM production mode parameters based on trial suggestions.

	Args:
		parameters (dict): Dictionary of hyperparameters.
		trial: Trial object for optimization.

	Returns:
		dict: Updated dictionary of hyperparameters.
	"""
	parameters.update(load_optuna_parameters('src/_settings/optimization_hyperparameters/lightgbm.json', 'gpu', trial))
	return parameters


# Function to generate LightGBM parameters for the 'long' configuration



# Function to generate SGDLinear parameters for the 'long' configuration
def sgdlinear_long_parameters(trial, loss, penalty, n_jobs, task_name):
	"""
	Generate SGDLinear parameters for the 'long' configuration based on trial suggestions.

	Args:
		trial: Trial object for optimization.
		loss (list): List of loss functions.
		penalty (list): List of penalty terms.
		n_jobs (int): Number of parallel jobs.
		task_name (str): Task name.

	Returns:
		dict: Dictionary of SGDLinear hyperparameters.
	"""
	parameters = load_optuna_parameters('src/_settings/optimization_hyperparameters/sgdlinear.json', 'long', trial)
	
	parameters.update(
			{
					'loss'   : trial.suggest_categorical('loss', loss),
					'penalty': trial.suggest_categorical('penalty', penalty)
					}
			)
	
	if task_name.startswith('classification'):
		parameters.update(load_optuna_parameters('src/_settings/optimization_hyperparameters/sgdlinear.json', 'classification', trial))
		parameters.update(
				{
						'n_jobs': trial.suggest_categorical('n_jobs', [n_jobs])
						}
				)
	
	return parameters


# Function to generate SGDLinear parameters for the 'short' configuration
def sgdlinear_short_parameters(trial, task_name):
	"""
	Generate SGDLinear parameters for the 'short' configuration based on trial suggestions.

	Args:
		trial: Trial object for optimization.
		task_name (str): Task name.

	Returns:
		dict: Dictionary of SGDLinear hyperparameters.
	"""
	return load_optuna_parameters('src/_settings/optimization_hyperparameters/sgdlinear.json', 'short', trial)
