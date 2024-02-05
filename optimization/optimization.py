import numpy as np
import optuna

from automl.settings import METRICS
from train.optimization_pruning import apply_pruning
from utils.functions import update_dict_with_new_keys
from utils.optimization import generate_hyperparameter_space


def optuna_optimization(data, cv, hyperparameters, n_trials, patience, optimization_type, n_jobs, task_name, model_name, metric_name, test_mode):
	x_data, y_data, weight_data = data
	
	best_score = []
	
	def objective(trial):
		space = generate_hyperparameter_space(task_name, model_name, metric_name, trial, optimization_type, len(np.unique(y_data)), n_jobs, test_mode)
		space = update_dict_with_new_keys(space, hyperparameters)
		
		mean_score, std_score = apply_pruning([x_data, y_data, weight_data], cv, space, best_score, patience, task_name, model_name, metric_name, trial)
		trial.set_user_attr('std_score', std_score)
		
		return mean_score
	
	return optimization_pipeline(metric_name, n_trials, hyperparameters, objective)


def optimization_pipeline(metric_name, n_trials, hyperparameters, objective_function):
	optimization_direction = METRICS[metric_name][0]
	
	study = optuna.create_study(direction=optimization_direction, sampler=optuna.samplers.TPESampler())
	study.optimize(objective_function, n_trials=n_trials)
	
	study_results = [(trial.values[0], trial.user_attrs.get('std_score'), trial.duration, trial.params) for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
	if len(study_results) == 0:
		return {}
	
	if optimization_direction == 'maximize':
		study_results.sort(key=lambda value: (-value[0], value[1], value[2]))
	
	else:
		study_results.sort(key=lambda value: (value[0], value[1], value[2]))
	
	best_hyperparameters = study_results[0][3]
	best_hyperparameters = update_dict_with_new_keys(best_hyperparameters, hyperparameters)
	
	return best_hyperparameters
