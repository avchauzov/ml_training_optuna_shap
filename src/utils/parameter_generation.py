"""
This module contains a function for generating hyperparameters based on task, model, and optimization type.
"""

from src.utils.space_generation import lightgbm_long_parameters, lightgbm_short_parameters, multinomialnb_parameters, set_lightgbm_production_mode_parameters, sgdlinear_long_parameters, sgdlinear_short_parameters


def generate_hyperparameters(task_name, model_name, metric_name, trial, optimization_type, num_class, n_jobs, test_mode):
	"""
	Generate hyperparameters based on task, model, and optimization type.

	Args:
		task_name (str): The name of the task.
		model_name (str): The name of the model.
		metric_name (str): The name of the metric.
		trial: Trial object for optimization.
		optimization_type (str): Type of optimization ('long' or 'short').
		num_class (int): Number of classes for multiclass tasks.
		n_jobs (int): Number of parallel jobs.
		test_mode (bool): Flag indicating test mode.

	Returns:
		dict: A dictionary of hyperparameters.
	"""
	# Define default parameters dictionary.
	parameters = {}
	
	# Check the task, model, and optimization type and generate hyperparameters accordingly.
	if task_name == 'classification_binary' and model_name == 'lightgbm':
		if optimization_type == 'long':
			objective = ['binary']
			metric = ['auc', 'average_precision', 'binary_logloss']
			parameters = lightgbm_long_parameters(trial, objective, metric, 1, n_jobs)
		elif optimization_type == 'short':
			parameters = lightgbm_short_parameters(trial)
	
	elif task_name == 'classification_multiclass' and model_name == 'lightgbm':
		if optimization_type == 'long':
			objective = ['multiclass'] if metric_name in ['roc_auc_ovr'] else ['multiclassova'] if metric_name in ['roc_auc_ovo'] else ['multiclass', 'multiclassova']
			metric = ['auc_mu', 'multi_logloss', 'multi_error']
			parameters = lightgbm_long_parameters(trial, objective, metric, num_class, n_jobs)
		elif optimization_type == 'short':
			parameters = lightgbm_short_parameters(trial)
	
	elif task_name == 'regression' and model_name == 'lightgbm':
		if optimization_type == 'long':
			objective = ['regression', 'regression_l1', 'huber', 'fair', 'quantile', 'mape']
			metric = ['l1', 'l2', 'rmse', 'quantile', 'mape', 'huber', 'fair', 'poisson', 'tweedie']
			parameters = lightgbm_long_parameters(trial, objective, metric, 1, n_jobs)
		elif optimization_type == 'short':
			parameters = lightgbm_short_parameters(trial)
	
	elif task_name == 'classification_binary' and model_name == 'sgdlinear':
		if optimization_type == 'long':
			loss = ['log_loss', 'modified_huber']
			penalty = ['l2', 'l1', 'elasticnet']
			parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, optimization_type)
		elif optimization_type == 'short':
			parameters = sgdlinear_short_parameters(trial, optimization_type)
	
	elif task_name == 'classification_multiclass' and model_name == 'sgdlinear':
		if optimization_type == 'long':
			loss = ['modified_huber']
			penalty = ['l2', 'l1', 'elasticnet']
			parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, optimization_type)
		elif optimization_type == 'short':
			parameters = sgdlinear_short_parameters(trial, optimization_type)
	
	elif task_name == 'regression' and model_name == 'sgdlinear':
		if optimization_type == 'long':
			loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
			penalty = ['l2', 'l1', 'elasticnet']
			parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, optimization_type)
		elif optimization_type == 'short':
			parameters = sgdlinear_short_parameters(trial, optimization_type)
	
	elif task_name == 'classification_multiclass' and model_name == 'multinomialnb':
		parameters = multinomialnb_parameters(trial)
	
	# Implement production mode for LightGBM if not in test mode.
	if not test_mode and model_name == 'lightgbm' and optimization_type == 'long':
		parameters = set_lightgbm_production_mode_parameters(parameters, trial)
	
	return parameters
