"""
This module contains functions for generating hyperparameters for different tasks, models, and optimization types.
"""


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
	parameters['device'] = trial.suggest_categorical('device', ['gpu'])
	parameters['gpu_use_dp'] = trial.suggest_categorical('gpu_use_dp', [False])
	return parameters


# Function to generate LightGBM parameters for the 'long' configuration
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
	parameters = {
			'num_leaves'       : trial.suggest_int('num_leaves', 2, 1024),
			'max_depth'        : trial.suggest_int('max_depth', 2, 16),
			'min_child_samples': trial.suggest_int('min_child_samples', 2, 1024),
			'max_bin'          : trial.suggest_int('max_bin', 16, 64),
			'min_data_in_bin'  : trial.suggest_int('min_data_in_bin', 2, 1024),
			'max_cat_threshold': trial.suggest_int('max_cat_threshold', 2, 128),
			'subsample_freq'   : trial.suggest_int('subsample_freq', 0, 1024),
			'n_estimators'     : trial.suggest_int('n_estimators', 128, 128),
			'extra_trees'      : trial.suggest_categorical('extra_trees', [True, False]),
			'boosting_type'    : trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
			'objective'        : trial.suggest_categorical('objective', objective),
			'metric'           : trial.suggest_categorical('metric', metric),
			'learning_rate'    : trial.suggest_categorical('learning_rate', [0.1]),
			'n_jobs'           : trial.suggest_categorical('n_jobs', [n_jobs]),
			'verbosity'        : trial.suggest_categorical('verbosity', [-1]),
			'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-6, 128.0, log=True),
			'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-6, 128.0, log=True),
			'min_split_gain'   : trial.suggest_float('min_split_gain', 1e-6, 128.0, log=True),
			'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.25, 1.0, step=0.05),
			'subsample'        : trial.suggest_float('subsample', 0.25, 0.95, step=0.05),
			'min_child_weight' : trial.suggest_float('min_child_weight', 1e-6, 1e6, log=True),
			'path_smooth'      : trial.suggest_float('path_smooth', 1e-6, 128.0, log=True),
			}
	
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
	return {
			'n_estimators' : trial.suggest_int('n_estimators', 64, 1024, step=1),
			'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, step=0.001),
			}


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
	parameters = {
			'loss'          : trial.suggest_categorical('loss', loss),
			'penalty'       : trial.suggest_categorical('penalty', penalty),
			'fit_intercept' : trial.suggest_categorical('fit_intercept', [True, False]),
			'early_stopping': trial.suggest_categorical('early_stopping', [False]),
			'verbose'       : trial.suggest_categorical('verbose', [0]),
			'alpha'         : trial.suggest_float('alpha', 1e-6, 128.0, log=True),
			'l1_ratio'      : trial.suggest_float('l1_ratio', 0.0, 1.0, step=0.01),
			'epsilon'       : trial.suggest_float('epsilon', 1e-6, 128.0, log=True),
			}
	
	if task_name.startswith('classification'):
		parameters.update(
				{
						'n_jobs'      : trial.suggest_categorical('n_jobs', [n_jobs]),
						'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
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
	parameters = {
			'max_iter'      : trial.suggest_int('max_iter', 512, 2048, step=1),
			'early_stopping': trial.suggest_categorical('early_stopping', [False]),
			}
	
	if task_name not in ['classification_multiclass']:
		parameters.update(
				{
						'n_iter_no_change'   : trial.suggest_int('n_iter_no_change', 2, 16, step=1),
						'validation_fraction': trial.suggest_float('validation_fraction', 0.05, 0.25, step=0.05),
						}
				)
	
	return parameters


# Function to generate parameters for MultinomialNB
def multinomialnb_parameters(trial):
	"""
	Generate parameters for MultinomialNB based on trial suggestions.

	Args:
		trial: Trial object for optimization.

	Returns:
		dict: Dictionary of MultinomialNB hyperparameters.
	"""
	return {
			'force_alpha': trial.suggest_categorical('force_alpha', [True, False]),
			'fit_prior'  : trial.suggest_categorical('fit_prior', [True, False]),
			'alpha'      : trial.suggest_float('alpha', 1e-6, 128.0, step=0.1),
			}
