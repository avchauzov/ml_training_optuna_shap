"""
This script contains functions for hyperparameter optimization using Optuna.
"""

import numpy as np
import optuna
import tqdm

from src.automl.metric_functions import METRIC_FUNCTIONS
from src.automl.pruning import apply_pruning
from src.utils.functions import update_dict_with_new_keys
from src.utils.parameter_generation import generate_hyperparameters


def optuna_hyperparameter_optimization(data, cv, hyperparameters, n_trials, patience, optimization_type, n_jobs, task_name, model_name, metric_name, test_mode):
	"""
	Perform hyperparameter optimization using Optuna.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		cv (list): List of cross-validation indices.
		hyperparameters (dict): Hyperparameters to be considered during optimization.
		n_trials (int): Number of trials for optimization.
		patience (int): Patience for early stopping during optimization.
		optimization_type (str): Type of optimization ('long' or 'short').
		n_jobs (int): Number of parallel jobs for optimization.
		task_name (str): Task name.
		model_name (str): Model name.
		metric_name (str): Metric name.
		test_mode (bool): Whether to run in test mode.

	Returns:
		dict: Best hyperparameters found during optimization.
	"""
	x_data, y_data, weight_data = data
	best_score = []
	
	def objective(trial):
		"""
		Objective function for Optuna hyperparameter optimization.
		
		Args:
			trial (optuna.Trial): The current trial to optimize.
		
		Returns:
			float: Mean score obtained during cross-validation.
		"""
		space = generate_hyperparameters(task_name, model_name, metric_name, trial, optimization_type, len(np.unique(y_data)), n_jobs, test_mode)
		space = update_dict_with_new_keys(space, hyperparameters)
		
		mean_score, std_score = apply_pruning([x_data, y_data, weight_data], cv, space, best_score, patience, task_name, model_name, metric_name, trial)
		trial.set_user_attr('std_score', std_score)
		
		return mean_score
	
	return optimize_with_study(metric_name, n_trials, hyperparameters, objective)


def optimize_with_study(metric_name, n_trials, hyperparameters, objective_function):
	"""
	Perform optimization using Optuna's study.

	Args:
		metric_name (str): Metric name.
		n_trials (int): Number of trials for optimization.
		hyperparameters (dict): Hyperparameters to be considered during optimization.
		objective_function (callable): Objective function for optimization.

	Returns:
		dict: Best hyperparameters found during optimization.
	"""
	optimization_direction = METRIC_FUNCTIONS[metric_name][0]
	study = optuna.create_study(direction=optimization_direction, sampler=optuna.samplers.TPESampler())
	
	with tqdm.tqdm(total=n_trials, position=0, dynamic_ncols=True) as pbar:
		def objective_function_with_progress_bar(trial):
			"""
			Objective function with a progress bar for Optuna hyperparameter optimization.
			
			Args:
			    trial (optuna.Trial): The current trial to optimize.

			Returns:
			    float: The score obtained during the trial.
			"""
			pbar.update()  # Update the progress bar for each trial
			
			try:
				best_value = trial.study.best_value
				pbar.set_postfix({metric_name: best_value})  # Update the best score in tqdm output
			
			except ValueError:
				pass
			
			return objective_function(trial)
		
		study.optimize(objective_function_with_progress_bar, n_trials=n_trials)
	
	study_results = [(trial.values[0], trial.user_attrs.get('std_score'), trial.duration, trial.params) for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
	if not study_results:
		return {}
	
	if optimization_direction == 'maximize':
		study_results.sort(key=lambda value: (-value[0], value[1], value[2]))
	else:
		study_results.sort(key=lambda value: (value[0], value[1], value[2]))
	
	best_hyperparameters = study_results[0][3]
	best_hyperparameters = update_dict_with_new_keys(best_hyperparameters, hyperparameters)
	
	return best_hyperparameters
