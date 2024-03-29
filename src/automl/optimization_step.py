"""
This script contains functions for optimizing machine learning models using Optuna,
including hyperparameter optimization and pruning strategies.
"""

import optuna

from src.models.cross_validation import calculate_cv_score


'''def apply_pruning(hyperparameters):
	if hyperparameters.get('num_leaves', 0) > (2 ** hyperparameters.get('max_depth', 0) * 0.75):
		return True
	
	return False'''

def perform_trial(data, cv, space, best_score_history, patience, task_name, model_name, metric_name, trial):
	"""
	Apply pruning to the optimization trial based on best score history.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		cv (list): List of cross-validation fold indices.
		space (dict): Hyperparameter space for optimization.
		best_score_history (list): List of best scores achieved during optimization.
		patience (int): Number of consecutive trials with no improvement before pruning.
		task_name (str): Task name.
		model_name (str): Model name.
		metric_name (str): Metric name.
		trial (optuna.Trial): The current trial to be pruned.

	Returns:
		tuple: Mean and standard deviation of test errors from cross-validation.
	"""
	x_data, y_data, weight_data = data
	
	try:
		best_value = trial.study.best_value
		best_score_history.append(best_value)
	except ValueError:
		pass
	
	if len(best_score_history) >= patience:
		if best_score_history[-1] == best_score_history[-patience]:
			raise optuna.TrialPruned()
	
	return calculate_cv_score([x_data, y_data, weight_data], cv, space, task_name, model_name, metric_name)
