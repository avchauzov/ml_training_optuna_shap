import numpy as np
import pandas as pd
import shap

from data.data_weights import split_and_weight_data
from tasks.functions import calculate_test_error
from train.train_models import train_lightgbm_model, train_sgdlinear_model


def select_important_features(x_data, train_error, test_error, to_use, metrics, scoring):
	"""
	Select important features based on error metrics and other criteria.

	Parameters:
	- to_use: List of candidate feature sets
	- x_data: Feature data
	- train_error: List of training error values
	- test_error: List of test error values
	- metrics: Dictionary mapping scoring methods to optimization directions
	- scoring: Scoring method (e.g., 'neg_mean_squared_error')

	Returns:
	- List of selected important features
	"""
	if len(to_use) == 0:
		return sorted(list(x_data))
	
	# Create a DataFrame to store error metrics and feature sets
	report_df = pd.DataFrame()
	report_df['train_error_mean'] = [np.nanmean(value) for value in train_error]
	report_df['train_error_std'] = [np.nanstd(value) for value in train_error]
	report_df['test_error_mean'] = [np.nanmean(value) for value in test_error]
	report_df['test_error_std'] = [np.nanstd(value) for value in test_error]
	report_df['columns_to_use'] = [value for value in to_use]
	report_df['len_columns_to_use'] = [len(value) for value in to_use]
	
	# Determine the optimization direction (minimize or maximize)
	direction = True if metrics.get(scoring[0]) == 'minimize' else False
	
	# Sort the DataFrame based on error metrics and feature set length
	report_df.sort_values(['test_error_mean', 'len_columns_to_use'], ascending=[direction, True], inplace=True)
	
	# Select the feature set with the best error metric
	columns_to_select_list = sorted(report_df['columns_to_use'].values[0])
	
	# Print the report DataFrame for reference (optional)
	print(report_df[['train_error_mean', 'test_error_mean', 'len_columns_to_use']])
	
	return columns_to_select_list


def process_data(task_name, model_name, x_data, y_data, weight_data, train_index, test_index, weight_adjustment, scoring, hyperparameters, train_error, test_error, index_column, x_values, shap_values):
	if model_name == 'sgdlinear':
		x_train, y_train, weight_train_list, metric_weight_train_list, x_test, y_test, weight_test_list, metric_weight_test_list = split_and_weight_data(x_data, y_data, train_index, weight_data)
		model = train_sgdlinear_model(x_train, y_train, weight_train_list, hyperparameters, task_name)
	
	elif model_name == 'lightgbm':
		x_train, y_train, weight_train_list, metric_weight_train_list, x_test, y_test, weight_test_list, metric_weight_test_list = split_and_weight_data(x_data, y_data, train_index, weight_data)
		model = train_lightgbm_model(x_train, y_train, weight_train_list, x_test, y_test, weight_test_list, hyperparameters, task_name)
	
	else:
		raise ValueError(f'Unknown model name: {model_name}')
	
	if task_name == 'regression':
		train_error.append(calculate_test_error(x_train, y_train, model, metric_weight_train_list, scoring))
		test_error.append(calculate_test_error(x_test, y_test, model, metric_weight_test_list, scoring))
	
	elif task_name == 'classification_binary':
		train_error.append(calculate_test_error(x_train, y_train, model, metric_weight_train_list, scoring))
		test_error.append(calculate_test_error(x_test, y_test, model, metric_weight_test_list, scoring))
	
	elif task_name == 'classification_multiclass':
		train_error.append(calculate_test_error(x_train, y_train, model, metric_weight_train_list, scoring))
		test_error.append(calculate_test_error(x_test, y_test, model, metric_weight_test_list, scoring))
	
	
	else:
		raise ValueError(f'Unknown task name: {task_name}')
	
	if model_name == 'sgdlinear':
		explainer = shap.LinearExplainer(model, x_train)
	
	else:
		explainer = shap.TreeExplainer(model)
	
	shap_sub_values = explainer.shap_values(x_test)
	
	if isinstance(shap_sub_values, list):
		shap_sub_values = [shap_sub_values[y_test][index] for index, y_test in enumerate(y_test)]
	
	index_column.extend(test_index)
	x_values.extend(x_test)
	shap_values.extend(shap_sub_values)


def get_select_features(x_data, y_data, weight_data, cv, hyperparameters, weight_adjustment, drop_rate, min_columns_to_keep, scoring, task_name, model, metric_dictionary):
	if x_data.shape[1] < min_columns_to_keep:
		return sorted(list(x_data))
	
	columns_to_select_list, columns_to_drop_flatten_list, iteration = list(x_data), [], 1
	train_error_column, test_error_column, columns_to_use_column, columns_to_drop_column = [], [], [], []
	while True:
		print(f'Iteration {iteration}')
		
		train_error, test_error = [], []
		index_column, x_values, shap_values = [], [], []
		for index, (train_index, test_index) in enumerate(cv):
			print('IterationCV {}/{}'.format(index + 1, len(cv)))
			process_data(task_name, model, x_data[columns_to_select_list], y_data, weight_data, train_index, test_index, weight_adjustment, scoring, hyperparameters, train_error, test_error, index_column, x_values, shap_values)
		
		train_error_column.append(train_error)
		test_error_column.append(test_error)
		
		columns_to_use_column.append(columns_to_select_list)
		
		#
		
		shap_values_df = pd.DataFrame(shap_values, columns=columns_to_select_list, index=index_column)
		shap_values_df = shap_values_df.groupby(shap_values_df.index).agg(np.nanmean)
		shap_values_df = np.abs(shap_values_df)
		shap_values_df = pd.concat(
				(pd.DataFrame(shap_values_df.std(axis=0).T, columns=['std_importance']),
				 pd.DataFrame(shap_values_df.mean(axis=0).T, columns=['mean_importance']),
				 pd.DataFrame(shap_values_df.max(axis=0).T, columns=['max_importance'])), axis=1
				)
		shap_values_df.sort_values(['std_importance', 'mean_importance', 'max_importance'], ascending=[False, False, False], inplace=True)
		
		columns_to_drop = shap_values_df.loc[(shap_values_df['std_importance'] == 0) & (shap_values_df['mean_importance'] == 0) & (shap_values_df['max_importance'] == 0)].index.tolist()
		shap_values_df = shap_values_df.loc[(shap_values_df['std_importance'] > 0) | (shap_values_df['mean_importance'] > 0) | (shap_values_df['max_importance'] > 0)]
		
		if shap_values_df.shape[0] == 0:
			break
		
		drop_index = shap_values_df.shape[0] - int(np.ceil(shap_values_df.shape[0] * drop_rate))
		
		if drop_index == 0:
			drop_index += 1
		
		columns_to_drop += shap_values_df[drop_index:].index.tolist()
		columns_to_select_list = shap_values_df[: drop_index].index.tolist()
		
		if len(columns_to_select_list) < min_columns_to_keep:
			break
		
		print(f'Not important columns: {len(columns_to_drop)}')
		print(columns_to_drop)
		
		print(f'Important columns: {len(columns_to_select_list)}')
		print(columns_to_select_list)
		
		columns_to_drop_flatten_list.extend(columns_to_drop)
		iteration += 1
	
	return select_important_features(x_data, train_error_column, test_error_column, columns_to_use_column, metric_dictionary, scoring)
