"""
This script contains functions for hyperparameter optimization using Optuna.
"""

import numpy as np
import optuna
import tqdm
from optuna.importance import get_param_importances
from optuna.trial import TrialState

from src._settings.metrics import METRICS
from src.automl.optimization_step import perform_trial
from src.data.preprocessing import preprocess_data
from src.utils.functions import update_dict
from src.utils.parameter_generation import generate_hyperparameters


def prepare_data(cv, data, model_name):
	x_data, y_data, weight_data = data
	
	train_test_set = []
	for train_index, test_index in cv:
		x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_data([x_data, y_data, weight_data], [train_index, test_index], model_name)
		train_test_set.append(([x_train, y_train, weight_train], [x_test, y_test, weight_test]))
	
	return train_test_set


def optimize_hyperparameters(data, cv, hyperparameters, n_trials, patience, optimization_type, n_jobs, task_name, model_name, metric_name, test_mode):
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
	train_test_set = prepare_data(cv, data, model_name)
	
	best_score = []
	
	def objective(trial):
		"""
		Objective function for Optuna hyperparameter optimization.
		
		Args:
			trial (optuna.Trial): The current trial to optimize.
		
		Returns:
			float: Mean score obtained during cross-validation.
		"""
		states_to_consider = (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)
		trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
		
		for _trial in reversed(trials_to_consider):
			if trial.params == _trial.params:
				return _trial.value
		
		space = generate_hyperparameters(task_name, model_name, metric_name, trial, optimization_type, len(np.unique(data[1])), n_jobs, test_mode)
		space = update_dict(space, hyperparameters)
		
		mean_score, std_score = perform_trial(train_test_set, cv, space, best_score, patience, task_name, model_name, metric_name, trial)
		trial.set_user_attr('std_score', std_score)
		
		return mean_score
	
	return run_study(metric_name, n_trials, hyperparameters, objective)


def run_study(metric_name, n_trials, hyperparameters, objective_function):
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
	optimization_direction = METRICS[metric_name][0]
	
	storage_url = 'sqlite:///study.db'
	study_name = 'study'
	
	try:
		optuna.delete_study(study_name=study_name, storage=storage_url)
	except KeyError:
		pass
	except Exception as error:
		raise f'An error occurred while deleting the study: {error}'
	
	study = optuna.create_study(direction=optimization_direction, sampler=optuna.samplers.TPESampler(), study_name=study_name, storage=storage_url, load_if_exists=False)
	
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
	
	try:
		parameter_importance = get_param_importances(study)
		parameter_importance = dict(sorted(parameter_importance.items(), key=lambda parameter: (-parameter[1], parameter[0])))
		
		print(f'Parameter importance: {parameter_importance}')
	except RuntimeError as error:
		print(f'Parameter importance: passing because of "{error}"')
	
	study_results = [(trial.values[0], trial.user_attrs.get('std_score'), trial.duration, trial.params) for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
	if not study_results:
		return {}
	
	if optimization_direction == 'maximize':
		study_results.sort(key=lambda value: (-value[0], value[1], value[2]))
	else:
		study_results.sort(key=lambda value: (value[0], value[1], value[2]))
	
	best_hyperparameters = study_results[0][3]
	best_hyperparameters = update_dict(best_hyperparameters, hyperparameters)
	
	optuna.delete_study(study_name=study_name, storage=storage_url)
	return best_hyperparameters
