"""
This module contains a function for generating hyperparameters based on task, model, and optimization type.
"""

from src.utils.space_generation import (
	elasticnet_long_parameters, elasticnet_short_parameters, lightgbm_long_parameters, lightgbm_short_parameters, logisticregression_long_parameters, logisticregression_short_parameters, multinomialnb_parameters, set_lightgbm_production_mode_parameters, sgdlinear_long_parameters,
	sgdlinear_short_parameters,
	)


def generate_hyperparameters(task_name, model_name, metric_name, trial, optimization_type, num_class, n_jobs, test_mode):
	parameter_generators = {
			('lightgbm', 'long')           : lambda: lightgbm_long_parameters(trial, get_objective(model_name, task_name, metric_name), get_metric(task_name), task_name, num_class, n_jobs),
			('lightgbm', 'short')          : lambda: lightgbm_short_parameters(trial),
			('sgdlinear', 'long')          : lambda: sgdlinear_long_parameters(trial, get_loss(model_name, task_name), ['l2', 'l1', 'elasticnet'], n_jobs, optimization_type),
			('sgdlinear', 'short')         : lambda: sgdlinear_short_parameters(trial),
			('elasticnet', 'long')         : lambda: elasticnet_long_parameters(trial),
			('elasticnet', 'short')        : lambda: elasticnet_short_parameters(trial),
			('logisticregression', 'long') : lambda: logisticregression_long_parameters(trial, n_jobs),
			('logisticregression', 'short'): lambda: logisticregression_short_parameters(trial),
			('multinomialnb', 'long')      : lambda: multinomialnb_parameters(trial),
			}
	
	generator = parameter_generators.get((model_name, optimization_type), lambda: {})
	parameters = generator()
	
	if not test_mode and model_name == 'lightgbm' and optimization_type == 'long':
		parameters = set_lightgbm_production_mode_parameters(parameters, trial)
	
	return parameters


def get_objective(model_name, task_name, metric_name):
	if model_name == 'lightgbm':
		if task_name == 'classification_binary':
			return ['binary']
		
		elif task_name == 'classification_multiclass':
			return ['multiclass'] if metric_name in ['roc_auc_ovo', 'roc_auc_ovr'] else ['multiclass', 'multiclassova']
		
		elif task_name == 'regression':
			return ['regression', 'regression_l1', 'huber', 'fair', 'quantile', 'mape']
	
	return []


def get_metric(task_name):
	if task_name == 'classification_binary':
		return ['auc', 'logloss', 'error']
	
	elif task_name == 'classification_multiclass':
		return ['auc_mu', 'multi_logloss', 'multi_error']
	
	elif task_name == 'regression':
		return ['l1', 'l2', 'rmse']
	
	return []


def get_loss(model_name, task_name):
	if model_name == 'sgdlinear':
		if task_name in ['classification_binary', 'classification_multiclass']:
			return ['log_loss', 'modified_huber']
		
		elif task_name == 'regression':
			return ['squared_error', 'huber']
	
	return []


'''def generate_hyperparameters(task_name, model_name, metric_name, trial, optimization_type, num_class, n_jobs, test_mode):
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
	if task_name in ['classification_binary'] and model_name in ['lightgbm']:
		if optimization_type in ['long']:
			objective = ['binary']
			metric = ['auc', 'average_precision', 'binary_logloss']
			parameters = lightgbm_long_parameters(trial, objective, metric, 1, n_jobs)
		elif optimization_type in ['short']:
			parameters = lightgbm_short_parameters(trial)
	
	elif task_name in ['classification_multiclass'] and model_name in ['lightgbm']:
		if optimization_type in ['long']:
			objective = ['multiclass'] if metric_name in ['roc_auc_ovo', 'roc_auc_ovr'] else ['multiclass', 'multiclassova']
			metric = ['auc_mu', 'multi_logloss', 'multi_error']
			parameters = lightgbm_long_parameters(trial, objective, metric, num_class, n_jobs)
		elif optimization_type in ['short']:
			parameters = lightgbm_short_parameters(trial)
	
	elif task_name in ['regression'] and model_name in ['lightgbm']:
		if optimization_type in ['long']:
			objective = ['regression', 'regression_l1', 'huber', 'fair', 'quantile', 'mape']
			metric = ['l1', 'l2', 'rmse', 'quantile', 'mape', 'huber', 'fair', 'poisson', 'tweedie']
			parameters = lightgbm_long_parameters(trial, objective, metric, 1, n_jobs)
		elif optimization_type in ['short']:
			parameters = lightgbm_short_parameters(trial)
	
	elif task_name == 'regression' and model_name == 'sgdlinear':
		if optimization_type == 'long':
			loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
			penalty = ['l2', 'l1', 'elasticnet']
			parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, optimization_type)
		elif optimization_type == 'short':
			parameters = sgdlinear_short_parameters(trial, optimization_type)
	
	elif task_name in ['classification_binary'] and model_name in ['sgdlinear']:
		if optimization_type in ['long']:
			loss = ['log_loss', 'modified_huber']
			penalty = ['l2', 'l1', 'elasticnet']
			parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, optimization_type)
		elif optimization_type in ['short']:
			parameters = sgdlinear_short_parameters(trial, optimization_type)
	
	elif task_name in ['classification_multiclass'] and model_name in ['sgdlinear']:
		if optimization_type in ['long']:
			loss = ['modified_huber']
			penalty = ['l2', 'l1', 'elasticnet']
			parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, optimization_type)
		elif optimization_type in ['short']:
			parameters = sgdlinear_short_parameters(trial, optimization_type)
	
	elif model_name in ['elasticnet']:
		if optimization_type in ['long']:
			parameters = elasticnet_long_parameters(trial)
		elif optimization_type in ['short']:
			parameters = elasticnet_short_parameters(trial)
	
	elif model_name in ['logisticregression']:
		if optimization_type in ['long']:
			parameters = logisticregression_long_parameters(trial, n_jobs)
		elif optimization_type in ['short']:
			parameters = logisticregression_short_parameters(trial)
	
	elif task_name in ['classification_multiclass'] and model_name in ['multinomialnb']:
		parameters = multinomialnb_parameters(trial)
	
	# Implement production mode for LightGBM if not in test mode.
	if not test_mode and model_name in ['lightgbm'] and optimization_type in ['long']:
		parameters = set_lightgbm_production_mode_parameters(parameters, trial)
	
	return parameters'''
