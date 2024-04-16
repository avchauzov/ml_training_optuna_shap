"""
This module contains functions for performing feature selection using SHAP values.
"""
import time

import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

from src._settings.metrics import METRICS
from src.automl.study import prepare_data
from src.data.preprocessing import scale_data
from src.models.training import train_any_model
from src.utils.metric_calculation import calculate_test_error


def calculate_shap_values(task_name, model_name, metric_name, data, selected, error, hyperparameters, x_values, shap_values):
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
	
	report_df = pd.DataFrame()
	report_df['mean_train_error'] = [np.nanmean(value) for value in train_error]
	report_df['mean_test_error'] = [np.nanmean(value) for value in test_error]
	report_df['std_test_error'] = [np.nanstd(value) for value in test_error]
	report_df['number_of_features'] = [len(value) for value in selected]
	
	direction = METRICS[metric_name][0] == 'minimize'
	report_df = report_df.sort_values(['mean_test_error', 'std_test_error', 'number_of_features', 'mean_train_error'], ascending=[direction, True, True, direction])
	
	table = report_df.to_string(index=False, header=True, justify='center')
	
	# Add dashes before and after the table
	dashes = '-' * len(table.split('\n')[0])
	formatted_table = f'{dashes}\n{table}\n{dashes}'
	
	print(formatted_table)
	
	return selected[report_df.index[0]]


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
						[train_set, test_set], selected,
						[train_error, test_error], hyperparameters, x_values, shap_values
						)
				_index.extend(test_set[3])
			
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
			
			important = sorted(shap_values_df.index.values[:-drop_count])
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
	return generate_report([train_error_cv, test_error_cv], selected_cv, metric_name)
