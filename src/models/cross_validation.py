"""
This script provides a function for performing cross-validation on machine learning models.
"""

import numpy as np

from src.data.preprocessing import scale_data
from src.models.training import train_any_model
from src.utils.metric_calculation import calculate_test_error


def calculate_cv_score(data, cv, hyperparameters, task_name, model_name, metric_name):
	"""
	Perform cross-validation and return the mean and standard deviation of the test errors.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		cv (list): List of cross-validation fold indices.
		hyperparameters (dict): Hyperparameters for the machine learning model.
		task_name (str): Task name.
		model_name (str): Model name.
		metric_name (str): Metric name.

	Returns:
		tuple: Mean and standard deviation of test errors.
	"""
	errors = []
	for train_set, test_set in data:
		x_train, y_train, weight_train, _ = train_set
		x_test, y_test, weight_test, _ = test_set
		
		if 'scaler' in hyperparameters:
			scaler_name = hyperparameters.get('scaler', None)
			del hyperparameters['scaler']
			
			x_train, x_test = scale_data([x_train, x_test], scaler_name)
		
		model = train_any_model(model_name, [x_train, y_train, weight_train, x_test, y_test, weight_test], hyperparameters, task_name)
		
		if model:
			errors.append(calculate_test_error([x_test, y_test, weight_test], model, metric_name, task_name))
	
	# Calculate and return the mean and standard deviation of test errors
	return np.nanmean(errors), np.nanstd(errors)
