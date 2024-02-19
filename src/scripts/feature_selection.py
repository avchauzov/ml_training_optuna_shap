"""
This module contains functions for performing feature selection using SHAP values.
"""
import time

import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

from src._settings.metrics import METRIC_FUNCTIONS
from src.data.loading import split_data
from src.models.training import train_lightgbm_model, train_sgdlinear_model
from src.utils.functions import calculate_test_error


def calculate_error_metrics(errors):
	"""
	Calculate error metrics.

	Args:
		errors (tuple): A tuple containing train and test errors.

	Returns:
		dict: A dictionary containing mean and standard deviation of train and test errors.
	"""
	train_error, test_error = errors
	return {
			'train_error_mean': np.nanmean(train_error),
			'train_error_std' : np.nanstd(train_error),
			'test_error_mean' : np.nanmean(test_error),
			'test_error_std'  : np.nanstd(test_error),
			}


def report_generation(error, selected, metric_name):
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
	
	report_df = pd.DataFrame()
	report_df['Train Error'] = [np.nanmean(value) for value in train_error]
	report_df['Test Error'] = [np.nanmean(value) for value in test_error]
	report_df['Number of Columns'] = [len(value) for value in selected]
	
	direction = True if METRIC_FUNCTIONS[metric_name][0] in ['minimize'] else False
	report_df = report_df.sort_values(['Test Error', 'Number of Columns', 'Train Error'], ascending=[direction, True, direction])
	
	columns_to_select_list = sorted(report_df.index)
	table = report_df.to_string(index=False, header=True, justify='center')
	
	# Add dashes before and after the table
	dashes = '-' * len(table.split('\n')[0])
	formatted_table = f'{dashes}\n{table}\n{dashes}'
	
	print(formatted_table)
	
	return selected[report_df.index[0]]


def shap_values_calculation(task_name, model_name, metric_name, data, index, error, hyperparameters, x_values, shap_values):
	"""
	Calculate SHAP values for feature selection.

	Args:
		task_name (str): Name of the task.
		model_name (str): Name of the model.
		metric_name (str): Name of the metric.
		data (list): List containing x_data, y_data, and weight_data.
		index (list): List containing train_index and test_index.
		error (list): List containing train_error and test_error.
		hyperparameters: Hyperparameters for the model.
		x_values (list): List to store input values.
		shap_values (list): List to store SHAP values.

	Returns:
		None
	"""
	x_data, y_data, weight_data = data
	train_index, test_index = index
	train_error, test_error = error
	
	x_train, y_train, weight_train, x_test, y_test, weight_test = split_data(
			[x_data, y_data, weight_data], [train_index, test_index]
			)
	
	if model_name == 'sgdlinear':
		model = train_sgdlinear_model([x_train, y_train, weight_train], hyperparameters, task_name)
	elif model_name == 'lightgbm':
		model = train_lightgbm_model(
				[x_train, y_train, weight_train], [x_test, y_test, weight_test],
				hyperparameters, task_name
				)
	else:
		raise ValueError(f'Unknown model name: {model_name}')
	
	train_error.append(calculate_test_error([x_train, y_train, weight_train], model, metric_name, task_name))
	test_error.append(calculate_test_error([x_test, y_test, weight_test], model, metric_name, task_name))
	
	if model_name == 'sgdlinear':
		explainer = shap.LinearExplainer(model, x_train)
	else:
		explainer = shap.TreeExplainer(model)
	
	shap_sub_values = explainer.shap_values(x_test)
	if isinstance(shap_sub_values, list):
		shap_sub_values = [shap_sub_values[y_test][index] for index, y_test in enumerate(y_test)]
	
	x_values.extend(x_test)
	shap_values.extend(shap_sub_values)


def feature_selection(data, cv, hyperparameters, drop_rate, min_columns_to_keep, task_name, model_name, metric_name):
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
	x_data, y_data, weight_data = data
	
	if x_data.shape[1] <= min_columns_to_keep:
		return sorted(list(x_data))
	
	selected_cv, to_drop_cv, iteration = [list(x_data)], [[]], 1
	train_error_cv, test_error_cv = [], []
	selected, to_drop = list(x_data), []
	
	with tqdm(total=len(selected) - min_columns_to_keep, position=0, dynamic_ncols=True) as pbar:
		while len(selected) >= min_columns_to_keep:
			train_error, test_error = [], []
			_index, x_values, shap_values = [], [], []
			
			for index, (train_index, test_index) in enumerate(cv):
				shap_values_calculation(
						task_name, model_name, metric_name,
						[x_data[selected], y_data, weight_data], [train_index, test_index],
						[train_error, test_error], hyperparameters, x_values, shap_values
						)
				_index.extend(test_index)
			
			train_error_cv.append(train_error)
			test_error_cv.append(test_error)
			
			if len(selected) == min_columns_to_keep:
				break
			
			shap_values_df = pd.DataFrame(shap_values, columns=selected, index=_index)
			shap_values_df = shap_values_df.groupby(shap_values_df.index).agg(np.nanmean)
			shap_values_df = np.abs(shap_values_df)
			
			shap_values_df = pd.concat(
					[
							pd.DataFrame(shap_values_df.std(axis=0).T, columns=['std_importance']),
							pd.DataFrame(shap_values_df.mean(axis=0).T, columns=['mean_importance']),
							pd.DataFrame(shap_values_df.max(axis=0).T, columns=['max_importance']),
							],
					axis=1
					)
			
			shap_values_df = shap_values_df.sort_values(
					['mean_importance', 'max_importance', 'std_importance'],
					ascending=[False, False, False]
					)
			drop_count = int(np.ceil(shap_values_df.shape[0] * drop_rate))
			
			important = sorted(shap_values_df.index.values[: -drop_count])
			not_important = sorted(shap_values_df.index.values[-drop_count:])
			
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
	return report_generation([train_error_cv, test_error_cv], selected_cv, metric_name)
