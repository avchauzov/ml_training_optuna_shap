"""
This script provides a function for performing cross-validation on machine learning models.
"""

import numpy as np

from src.data.preprocessing import preprocess_data
from src.models.training import train_elasticnet_model, train_lightgbm_model, train_logisticregression_model, train_multinomialnb_model, train_sgdlinear_model
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
	x_data, y_data, weight_data = data
	errors = []
	
	for train_index, test_index in cv:
		scaler_name = hyperparameters.get('scaler', None)
		
		if 'scaler' in hyperparameters:
			del hyperparameters['scaler']
		
		# Preprocess data and split it into training and test sets
		x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_data([x_data, y_data, weight_data], [train_index, test_index], scaler_name, model_name)
		
		# Train the model based on the model name
		if model_name == 'lightgbm':
			model = train_lightgbm_model([x_train, y_train, weight_train], [x_test, y_test, weight_test], hyperparameters, task_name)
		elif model_name == 'multinomialnb':
			model = train_multinomialnb_model([x_train, y_train, weight_train], hyperparameters)
		elif model_name == 'sgdlinear':
			model = train_sgdlinear_model([x_train, y_train, weight_train], hyperparameters, task_name)
		elif model_name == 'elasticnet':
			model = train_elasticnet_model([x_train, y_train, weight_train], hyperparameters, task_name)
		elif model_name == 'logisticregression':
			model = train_logisticregression_model([x_train, y_train, weight_train], hyperparameters, task_name)
		else:
			model = None
		
		if model:
			errors.append(calculate_test_error([x_test, y_test, weight_test], model, metric_name, task_name))
	
	# Calculate and return the mean and standard deviation of test errors
	return np.nanmean(errors), np.nanstd(errors)
