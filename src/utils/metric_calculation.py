"""
This module contains functions for calculating test errors and metrics, as well as updating dictionaries.
"""

from src._settings.metrics import METRICS


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
	
	if task_name in ['classification_binary']:
		predictions = predictions[:, 1]
	
	metric_function = METRICS[metric_name][1]
	return metric_function(y_true=y_test, y_pred=predictions, sample_weight=sample_weight_test)
