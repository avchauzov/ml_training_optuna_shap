import numpy as np
import optuna
from tasks.classification_binary import calculate_prediction_error as classification_binary_calculate_prediction_error
from tasks.classification_multiclass import calculate_prediction_error as classification_multiclass_calculate_prediction_error
from tasks.regression import calculate_prediction_error as regression_calculate_prediction_error

from models._lightgbm import split_and_weight_data as lightgbm_split_and_weight_data, train_model as lightgbm_train_model
from models._multinomialnb import train_model as multinomialnb_train_model
from models._sgdlinear import split_and_weight_data as sgdmodel_split_and_weight_data, train_model as sgdmodel_train_model


def cross_validate(x_data, y_data, weight_data, cv, hyperparameters, weight_adjustment, scoring, task_name, model_name):
	"""
	Perform cross-validation and compute error scores based on specified conditions and models.

	Parameters:
	- x_data: Feature data
	- y_data: True labels
	- cv: Cross-validation iterator
	- hyperparameters: Model hyperparameters
	- weighing: Flag indicating whether to adjust weights
	- scoring: Scoring method (e.g., 'weighted')
	- task_name: Machine learning task (e.g., 'regression', 'classification_binary', 'classification_multiclass')
	- model_name: Machine learning model (e.g., 'sgdregressor', 'sgdclassifier', 'lightgbm', 'multinomialnb', 'sgdlinear')

	Returns:
	- Mean error score across cross-validation folds
	"""
	error = []
	for train_index, test_index in cv:
		
		if model_name == 'sgdlinear':
			x_train, y_train, weight_train, weight_metric_train, x_test, y_test, weight_test, weight_metric_test = sgdmodel_split_and_weight_data(x_data, y_data, weight_data, train_index, test_index, weight_adjustment, scoring, task_name)
			model = sgdmodel_train_model(x_train, y_train, weight_train, hyperparameters, task_name)
		
		elif model_name == 'lightgbm':
			x_train, y_train, weight_train, weight_metric_train, x_test, y_test, weight_test, weight_metric_test = lightgbm_split_and_weight_data(x_data, y_data, weight_data, train_index, test_index, weight_adjustment, scoring)
			model = lightgbm_train_model(x_train, y_train, weight_train, x_test, y_test, weight_test, hyperparameters, task_name)
		
		elif model_name == 'multinomialnb':
			x_train, y_train, weight_train, weight_metric_train, x_test, y_test, weight_test, weight_metric_test = sgdmodel_split_and_weight_data(x_data, y_data, weight_data, train_index, test_index, weight_adjustment, scoring, task_name)
			model = multinomialnb_train_model(x_train, y_train, weight_train, hyperparameters)
		
		if task_name == 'regression':
			error.append(regression_calculate_prediction_error(x_test, y_test, model, weight_metric_test, scoring))
		
		elif task_name == 'classification_binary':
			error.append(classification_binary_calculate_prediction_error(x_test, y_test, model, weight_metric_test, scoring))
		
		elif task_name == 'classification_multiclass':
			error.append(classification_multiclass_calculate_prediction_error(x_test, y_test, model, weight_metric_test, scoring))
	
	return np.nanmean(error), np.nanstd(error)


def apply_pruning(x_data, y_data, weight_data, cv, trial, space, scoring, best_score_list, patience, task_name, model_name):
	"""
	Apply pruning during an Optuna hyperparameter optimization process.

	Parameters:
	- trial: Optuna trial object
	- best_score_list: List to store the best scores during optimization
	- patience: Number of consecutive trials with no improvement before pruning
	- space: Hyperparameter space to optimize
	- x_data: Feature data
	- y_data: True labels
	- cv: Cross-validation iterator
	- scoring: Scoring method (e.g., 'weighted')
	- task_name: Machine learning task (e.g., 'regression', 'classification_binary', 'classification_multiclass')
	- model_name: Machine learning model (e.g., 'sgdregressor', 'sgdclassifier', 'lightgbm', 'multinomialnb', 'sgdlinear')

	Returns:
	- Cross-validation score with applied pruning
	"""
	try:
		best_value = trial.study.best_value
		best_score_list.append(best_value)
	except ValueError as _:
		pass
	
	if len(best_score_list) >= patience:
		if best_score_list[-1] == best_score_list[-patience]:
			raise optuna.TrialPruned()
	
	weight_adjustment = space.pop('weight_adjustment', None)
	
	return cross_validate(x_data, y_data, weight_data, cv, space, weight_adjustment, scoring, task_name, model_name)


def perform_optuna_optimization(objective_function, hyperparameters, metrics, scoring, n_trials):
	"""
	Perform hyperparameter optimization using Optuna.

	Parameters:
	- metrics: Dictionary mapping scoring methods to optimization directions
	- scoring: Scoring method (e.g., 'neg_mean_squared_error')
	- n_trials: Number of optimization trials
	- hyperparameters: Initial hyperparameters for optimization
	- objective_function: Objective function to optimize

	Returns:
	- Best hyperparameters and optional weight adjustment
	"""
	# Create an Optuna study with the specified optimization direction
	study = optuna.create_study(direction=metrics.get(scoring[0]), sampler=optuna.samplers.TPESampler())
	
	# Perform optimization with the provided objective function and number of trials
	study.optimize(objective_function, n_trials=n_trials)
	
	# Retrieve the best hyperparameters from the study
	study_results = [(trial.values[0], trial.user_attrs.get('std_score'), trial.duration, trial.params) for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
	
	# Check if no hyperparameters were found, return an empty dictionary and None
	if len(study_results) == 0:
		return {}, None
	
	if metrics.get(scoring[0]) == 'maximize':
		study_results.sort(key=lambda value: (-value[0], value[1], value[2]))
	
	else:
		study_results.sort(key=lambda value: (value[0], value[1], value[2]))
	
	best_hyperparameters = study_results[0][3]
	
	# Merge the best hyperparameters with the initial hyperparameters dictionary
	best_hyperparameters.update({key: value for key, value in hyperparameters.items() if key not in best_hyperparameters})
	
	# Pop the 'weight_adjustment' hyperparameter (if it exists) and return
	weight_adjustment = best_hyperparameters.pop('weight_adjustment', None)
	return best_hyperparameters, weight_adjustment
