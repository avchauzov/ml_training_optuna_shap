import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def symmetrical_mape(y_list, prediction_list, sample_weight=None):
	"""
	Calculate the Symmetrical Mean Absolute Percentage Error (sMAPE).

	Parameters:
	- y_list: Actual values
	- prediction_list: Predicted values
	- sample_weight: Optional list of weights for each data point (default is None)

	Returns:
	- Symmetrical Mean Absolute Percentage Error (sMAPE)
	"""
	if sample_weight is None:
		sample_weight = np.ones(len(y_list))  # Default weights are all ones
	
	numerator = np.nansum(np.multiply(sample_weight, np.abs(y_list - prediction_list)))
	denominator = np.nansum(np.multiply(sample_weight, np.abs(y_list) + np.abs(prediction_list)))
	
	if denominator == 0:
		return 0.0  # Handle division by zero case
	
	return 1 / len(y_list) * numerator / denominator


def calculate_error(y_list, prediction_list, weight_list=None, scoring=None):
	"""
	Calculate error based on the specified scoring method.

	Parameters:
	- prediction_list: Predicted values
	- y_list: Actual values
	- weight_list: Optional list of weights for each data point (default is None)
	- scoring: Scoring method (e.g., 'neg_mean_absolute_error', 'neg_mean_squared_error', 'sMAPE')

	Returns:
	- Calculated error using the specified scoring method
	"""
	if scoring is None:
		return np.nan  # Handle the case of undefined scoring
	
	if weight_list is None:
		weight_list = np.ones(len(y_list))  # Default weights are all ones
	
	if scoring[0] == 'neg_mean_absolute_error':
		return mean_absolute_error(y_list, prediction_list, sample_weight=weight_list)
	
	elif scoring[0] == 'neg_mean_squared_error':
		return mean_squared_error(y_list, prediction_list, sample_weight=weight_list)
	
	elif scoring[0] == 'sMAPE':
		return symmetrical_mape(y_list, prediction_list, sample_weight=weight_list)
	
	return np.nan  # Handle unknown or unsupported scoring methods


def calculate_prediction_error(x_test, y_test, model, sample_weight=None, scoring=None):
	"""
	Calculate the error of predictions made by the model on test data.

	Parameters:
	- x_test: Test feature data
	- y_test: Test target data
	- model: Trained machine learning model
	- sample_weight: Optional list of weights for each test data point (default is None)
	- scoring: Scoring method (e.g., 'neg_mean_absolute_error', 'neg_mean_squared_error', 'sMAPE')

	Returns:
	- Calculated error of the model's predictions on the test data
	"""
	if scoring is None:
		return np.nan  # Handle the case of undefined scoring
	
	if sample_weight is None:
		sample_weight = np.ones(len(y_test))  # Default weights are all ones
	
	prediction_list = model.predict(x_test)
	
	return calculate_error(y_test, prediction_list, sample_weight, scoring)


def compute_sample_weights(x_train, y_train, x_test, y_test, weighing, scoring):
	"""
	Calculate sample weights for training and testing data based on specified parameters.

	Parameters:
	- x_train: Training feature data
	- y_train: Training target data
	- x_test: Test feature data
	- y_test: Test target data
	- weighing: Flag indicating whether to adjust weights
	- scoring: Scoring method (e.g., ('sMAPE', 'weighted'))

	Returns:
	- Training and testing data along with calculated sample weights
	"""
	weight_train_list, weight_test_list = np.ones(len(y_train)), np.ones(len(y_test))
	metric_weight_train_list, metric_weight_test_list = np.ones(len(y_train)), np.ones(len(y_test))
	
	"""
	"""
	
	return x_train, y_train, weight_train_list, metric_weight_train_list, x_test, y_test, weight_test_list, metric_weight_test_list
