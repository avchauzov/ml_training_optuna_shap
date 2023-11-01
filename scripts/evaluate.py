import numpy as np

from scripts.train import apply_pruning, perform_optuna_optimization
from utils.optimization_utils import generate_hyperparameter_space


def optimize_hyperparameters(x_data, y_data, cv, hyperparameters, n_jobs, n_trials, patience, scoring, optimization_type, task_name, model_name, metrics):
	"""
	Perform hyperparameter optimization for a machine learning model.

	Parameters:
	- x_data: Feature data
	- y_data: Target labels
	- cv: Cross-validation iterator
	- hyperparameters: Dictionary of hyperparameters to optimize
	- n_jobs: Number of parallel jobs for optimization
	- n_trials: Number of optimization trials
	- patience: Patience for early stopping during optimization
	- scoring: Scoring method (e.g., 'neg_mean_squared_error')
	- optimization_type: Type of optimization space (e.g., 'long', 'short')
	- task_name: Machine learning task (e.g., 'regression', 'classification_binary', 'classification_multiclass')
	- model_name: Machine learning model (e.g., 'sgdregressor', 'sgdclassifier', 'lightgbm', 'multinomialnb', 'sgdlinear')
	- metrics: Dictionary mapping scoring methods to optimization directions

	Returns:
	- Best hyperparameters and optional weight adjustment
	"""
	best_score_list = []
	
	def objective(trial):
		# Generate hyperparameter space for the current trial
		space = generate_hyperparameter_space(task_name, model_name, optimization_type, trial, scoring, len(np.unique(y_data)), n_jobs)
		
		# Merge trial-specific hyperparameters with the provided ones
		for key in hyperparameters.keys():
			if key not in space:
				space[key] = hyperparameters[key]
		
		# Apply pruning strategy and return the cross-validation score
		return apply_pruning(x_data, y_data, cv, trial, space, scoring, best_score_list, patience, task_name, model_name)
	
	# Perform Optuna hyperparameter optimization
	return perform_optuna_optimization(objective, hyperparameters, metrics, scoring, n_trials)


def is_model_pruned(model_type, target_data, hyperparameters):
	"""
	Checks if the given model should be pruned based on specific criteria.

	Args:
		model_type (str): The type of the model (e.g., 'lightgbm', 'sgdclassifier').
		hyperparameters (dict): A dictionary containing hyperparameters relevant to the model.
		target_data (array-like): The target data used for training the model.

	Returns:
		bool: True if the model should be pruned; False otherwise.
	"""
	if (
			model_type == 'lightgbm' and (
			hyperparameters['num_leaves'] >= (2 ** hyperparameters['max_depth']) * 0.75 or
			(hyperparameters['subsample_freq'] == 0 and hyperparameters['subsample'] > 0.0) or
			(hyperparameters['subsample_freq'] > 0 and hyperparameters['subsample'] == 0.0) or
			((hyperparameters['subsample_freq'] > 0 or hyperparameters['subsample'] > 0.0) and
			 hyperparameters['boosting_type'] == 'goss') or
			(hyperparameters['objective'] in ['poisson', 'gamma', 'tweedie'] and np.nanmin(target_data) < 0)
	)
	):
		return True
	
	if (
			model_type == 'sgdclassifier' and
			hyperparameters['class_weight'] == 'balanced' and
			hyperparameters['weight_adjustment']
	):
		return True
	
	return False
