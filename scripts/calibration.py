import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from models._lightgbm import split_and_weight_data as functions_lightgbm_split_and_weight_data, train_model as functions_lightgbm_train_model
from tasks.classification_binary import calculate_prediction_error as functions_classification_binary_calculate_prediction_error
from tasks.classification_multiclass import calculate_prediction_error as functions_classification_multiclass_calculate_prediction_error


def get_calibration_error(_x_data, _y_data, weight_data, _cv, _hyperparameters_dictionary, _weight_adjustment, _n_jobs, _scoring, _model_type, _space, _task):
	error_list = []
	for train_index, test_index in _cv:
		x_train, y_train, weight_train_list, metric_weight_train_list, x_test, y_test, weight_test_list, metric_weight_test_list = functions_lightgbm_split_and_weight_data(_x_data, _y_data, weight_data, train_index, test_index, _weight_adjustment, _scoring, _task)
		model = functions_lightgbm_train_model(x_train, y_train, weight_train_list, x_test, y_test, weight_test_list, _hyperparameters_dictionary, _task)
		
		if _model_type == 'LogisticRegression':
			calibration_model = LogisticRegression(penalty=_space['penalty'], C=_space['C'], fit_intercept=_space['fit_intercept'], class_weight=_space['class_weight'], solver=_space['solver'], n_jobs=_space['n_jobs'])
			
			train_leaves = model.predict_proba(x_train, pred_leaf=True)
			test_leaves = model.predict_proba(x_test, pred_leaf=True)
			
			if train_leaves.shape[1] != test_leaves.shape[1]:
				raise optuna.TrialPruned()
			
			encoder = OneHotEncoder()
			train_leaves = encoder.fit_transform(train_leaves)
			test_leaves = encoder.transform(test_leaves)
			
			if train_leaves.shape[1] != test_leaves.shape[1]:
				raise optuna.TrialPruned()
			
			calibration_model.fit(train_leaves, y_train, sample_weight=weight_train_list)
			prediction_list = calibration_model.predict_proba(test_leaves)
		
		else:
			prediction_list = model.predict_proba(x_test)
		
		if _task == 'classification_binary':
			error_list.append(functions_classification_binary_calculate_prediction_error(x_train, y_train, model, metric_weight_train_list, _scoring))
		
		elif _task == 'classification_multiclass':
			error_list.append(functions_classification_multiclass_calculate_prediction_error(x_train, y_train, model, metric_weight_train_list, _scoring))
	
	return np.nanmean(error_list)


def get_calibration_model(x_data, y_data, weight_data, cv, hyperparameters_dictionary, weight_adjustment, n_jobs, n_trials, scoring, metric_dictionary, task):
	"""
	Get the best calibration model based on optimization trials.

	Args:
		x_data (array-like): The input features for training.
		y_data (array-like): The target variable for training.
		cv (int): Number of cross-validation folds.
		hyperparameters_dictionary (dict): Dictionary of hyperparameters for the main model.
		weight_adjustment (bool): Flag indicating whether weight adjustment is applied.
		n_jobs (int): Number of CPU cores to use for training.
		n_trials (int): Number of optimization trials.
		scoring (str): The scoring metric used for optimization.
		metric_dictionary (dict): Dictionary of metric names and their corresponding directions.
		task (str): The machine learning task type (e.g., 'classification_binary', 'regression').

	Returns:
		tuple: A tuple containing the best calibration model name and its hyperparameters.
	"""
	results_dictionary = {
			'none': [get_calibration_error(
					x_data, y_data, weight_data, cv, hyperparameters_dictionary, weight_adjustment, n_jobs,
					scoring, 'none', {}, task
					), {}]
			}
	
	def objective(trial):
		space = {
				'penalty'      : trial.suggest_categorical('penalty', ['l1', 'l2']),
				'C'            : trial.suggest_float('C', 1e-6, 10.0, log=True),
				'fit_intercept': trial.suggest_categorical('fit_intercept', [False]),
				'class_weight' : trial.suggest_categorical('class_weight', [None, 'balanced']),
				'solver'       : trial.suggest_categorical('solver', ['saga']),
				'n_jobs'       : trial.suggest_categorical('n_jobs', [n_jobs])
				}
		
		return get_calibration_error(
				x_data, y_data, weight_data, cv, hyperparameters_dictionary, weight_adjustment, n_jobs,
				scoring, 'LogisticRegression', space, task
				)
	
	study = optuna.create_study(direction=metric_dictionary.get(scoring[0]), sampler=optuna.samplers.TPESampler())
	study.optimize(objective, n_trials=n_trials)
	
	results_dictionary['LogisticRegression'] = [study.best_value, study.best_params]
	results_dictionary = [[key] + value for key, value in results_dictionary.items()]
	
	reverse = False if metric_dictionary.get(scoring[0]) == 'minimize' else True
	results_dictionary.sort(key=lambda value: value[1], reverse=reverse)
	
	return results_dictionary[0][0], results_dictionary[0][2]
