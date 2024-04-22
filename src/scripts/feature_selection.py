"""
This module contains functions for performing feature selection using SHAP values.
"""
import time

import numpy as np
import shap
from tqdm import tqdm

from src._settings.metrics import METRICS
from src.automl.study import prepare_data
from src.data.preprocessing import scale_data
from src.models.training import train_any_model
from src.utils.metric_calculation import calculate_test_error


def calculate_shap_values(task_name, model_name, metric_name, data, error, hyperparameters, x_values, shap_values):
	train_set, test_set = data
	x_train, y_train, weight_train, index_train = train_set
	x_test, y_test, weight_test, index_test = test_set
	train_error, test_error = error
	
	if 'scaler' in hyperparameters:
		scaler_name = hyperparameters.get('scaler', None)
		del hyperparameters['scaler']
		
		x_train, x_test = scale_data([x_train, x_test], scaler_name)
	
	model = train_any_model(model_name, [x_train, y_train, weight_train, x_test, y_test, weight_test], hyperparameters, task_name)
	
	train_error.append(calculate_test_error([x_train, y_train, weight_train], model, metric_name, task_name))
	test_error.append(calculate_test_error([x_test, y_test, weight_test], model, metric_name, task_name))
	
	if model_name in ['sgdlinear', 'elasticnet', 'logisticregression']:
		explainer = shap.LinearExplainer(model, x_train)
	else:
		explainer = shap.TreeExplainer(model)
	
	shap_sub_values = explainer.shap_values(x_test)
	if isinstance(shap_sub_values, list):
		shap_sub_values = [shap_sub_values[y_test][index] for index, y_test in enumerate(y_test)]
	
	x_values.extend(x_test)
	shap_values.extend(shap_sub_values)


def generate_report(error, selected, metric_name):
	"""
	Generate a report based on errors, selected columns, and metric name.

	Args:
		error (list): A list containing train and test errors.
		selected (list): List of selected columns.
		metric_name (str): Name of the metric.

	Returns:
		list: Sorted list of selected columns.
	"""
	train_error, test_error = error
	
	mean_train_error = [np.nanmean(value) for value in train_error]
	mean_test_error = [np.nanmean(value) for value in test_error]
	std_test_error = [np.nanstd(value) for value in test_error]
	number_of_features = [len(value) for value in selected]
	
	direction = METRICS[metric_name][0] == 'minimize'
	sorted_indices = np.lexsort(
			(
					mean_train_error,
					number_of_features,
					std_test_error,
					mean_test_error if direction else -mean_test_error,
					)
			)
	
	table_rows = [
			f'{mean_train_error[i]:.4f} | {mean_test_error[i]:.4f} | {std_test_error[i]:.4f} | {number_of_features[i]}'
			for i in sorted_indices
			]
	table_header = 'Mean Train Error | Mean Test Error | Std Test Error | Number of Features'
	formatted_table = f'{table_header}\n{"-" * len(table_header)}\n'
	formatted_table += '\n'.join(table_rows)
	
	print(formatted_table)
	
	return selected[sorted_indices[0]]


def select_important_features(data, cv, hyperparameters, drop_rate, min_columns_to_keep, task_name, model_name, metric_name):
	"""
	Perform feature selection using SHAP values.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		cv (list): List of train-test indices.
		hyperparameters: Hyperparameters for the model.
		drop_rate (float): Rate of columns to drop in each iteration.
		min_columns_to_keep (int): Minimum number of columns to keep.
		task_name (str): Name of the task.
		model_name (str): Name of the model.
		metric_name (str): Name of the metric.

	Returns:
		list: Sorted list of selected columns.
	"""
	train_test_set = prepare_data(cv, data, model_name)
	
	if data[0].shape[1] <= min_columns_to_keep:
		return sorted(list(data[0]))
	
	selected_cv, to_drop_cv, iteration = [list(data[0])], [[]], 1
	train_error_cv, test_error_cv = [], []
	selected, to_drop = list(data[0]), []
	
	with tqdm(total=len(selected) - min_columns_to_keep, position=0, dynamic_ncols=True) as pbar:
		while len(selected) >= min_columns_to_keep:
			train_error, test_error = [], []
			_index, x_values, shap_values = [], [], []
			
			for index, (train_set, test_set) in enumerate(train_test_set):
				calculate_shap_values(
						task_name, model_name, metric_name,
						[train_set, test_set], [train_error, test_error], hyperparameters, x_values, shap_values
						)
				_index.extend(test_set[3])
			
			train_error_cv.append(train_error)
			test_error_cv.append(test_error)
			
			if len(selected) == min_columns_to_keep:
				break
			
			shap_values_mean = np.abs(shap_values).mean(axis=0)
			shap_values_std = np.abs(shap_values).std(axis=0)
			
			sorted_indices = np.argsort(
					(-shap_values_mean, -np.max(shap_values_mean, axis=0), -shap_values_std), axis=0
					)
			sorted_indices = np.flip(sorted_indices, axis=0)
			
			drop_count = int(np.ceil(sorted_indices.shape[0] * drop_rate))
			
			important = sorted_indices[:-drop_count]
			not_important = sorted_indices[-drop_count:]
			
			selected_cv.append(important)
			to_drop_cv.append(not_important)
			
			selected = important.copy()
			to_drop.extend(not_important)
			iteration += 1
			
			# Update the progress bar for each feature selection step
			pbar.update(len(not_important))
			pbar.set_postfix({'Selected Columns': len(selected)})
	
	time.sleep(1)
	print()  # Print a newline after completing the progress bar
	return generate_report([train_error_cv, test_error_cv], selected_cv, metric_name)
