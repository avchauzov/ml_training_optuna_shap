"""
This module contains functions for calculating test errors and metrics, as well as updating dictionaries.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.automl.metrics import METRIC_FUNCTIONS


def calculate_test_error(data, model, metric_name, task_name):
	"""
	Calculate the test error using the specified model and metric.

	Args:
		data (list): List containing x_test, y_test, and sample_weight_test.
		model: Trained machine learning model.
		metric_name (str): Metric name.
		task_name (str): Task name.

	Returns:
		float: Test error based on the specified metric.
	"""
	x_test, y_test, sample_weight_test = data
	
	if task_name in ['regression']:
		predictions = model.predict(x_test)
	else:
		predictions = model.predict_proba(x_test)
	
	# Calculate error using calculate_error function
	return calculate_error([y_test, predictions, sample_weight_test], task_name, metric_name)


def calculate_error(data, task_name, metric_name):
	"""
	Calculate the error using true and predicted values, considering the task type.

	Args:
		data (list): List containing y_true, y_pred, and sample_weight.
		task_name (str): Task name.
		metric_name (str): Metric name.

	Returns:
		float: Calculated error based on the specified metric and task.
	"""
	y_true, y_pred, sample_weight = data
	
	if task_name not in ['regression'] and y_true.ndim == 1:
		y_true = OneHotEncoder(sparse_output=False).fit_transform(np.array(y_true).reshape(-1, 1))
	
	# Calculate metric using calculate_metric function
	return calculate_metric([y_true, y_pred, sample_weight], metric_name)


def calculate_metric(data, metric_name):
	"""
	Calculate the metric based on the specified metric name.

	Args:
		data (list): List containing y_true, y_pred, and sample_weight.
		metric_name (str): Metric name.

	Returns:
		float: Calculated metric value.
	"""
	y_true, y_pred, sample_weight = data
	metric_function = METRIC_FUNCTIONS[metric_name][1]
	return metric_function(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


def update_dict_with_new_keys(space, hyperparameters):
	"""
	Update a dictionary with new keys and values from another dictionary.

	Args:
		space (dict): The dictionary to be updated.
		hyperparameters (dict): The dictionary containing new keys and values.

	Returns:
		dict: The updated dictionary.
	"""
	space.update((key, value) for key, value in hyperparameters.items() if key not in space)
	return space
